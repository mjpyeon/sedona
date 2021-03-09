import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import shutil
import os
import io
from numpy.linalg import svd
from collections import OrderedDict
from copy import deepcopy
from torch.optim.lr_scheduler import _LRScheduler, ReduceLROnPlateau

from config import config

class AverageMeter(object):
    """
    Computes and stores the average and
    current value.
    """
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

def get_error(output, target, topk=(1,)):
    """Computes the error over the k top predictions for the specified values of k"""
    with torch.no_grad():
        maxk = max(topk)
        batch_size = target.size(0)

        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))

        res = []
        for k in topk:
            correct_k = correct[:k].view(-1).float().sum(0, keepdim=True)
            res.append(1 - correct_k.mul_(1. / batch_size))
        return res

def move_model_state_dict(state_dict, device):
    for k, v in state_dict.items():
        state_dict[k] = v.to(device)

    return state_dict

def get_weight_config_str():
    weight_config = {
        'feat_mult': config.feat_mult,
        'dataset': config.dataset,
        'step': config.train_iters,
        'optim': config.optim,
        'lr': config.lr,
        'lr_min': config.lr_min,
        'smoothing': config.smoothing,
        'batch_size': config.batch_size,
        'valid_batch_size': config.valid_batch_size
    }
    if config.no_beta:
        weight_config['decoder'] = config.baseline_dec_type

    weight_config_str = '-'.join(['{}_{}'.format(k, get_formatted_string(v)) for k, v in weight_config.items()])
    return weight_config_str

def get_formatted_string(value):
    if isinstance(value, float):
        if value == 0.:
            formatted = '0'
        elif value < 1e-2 or value > 1e2:
            formatted = '{:.2e}'.format(value)
        else:
            formatted = '{:.4f}'.format(value)
    else:
        formatted = str(value)

    return formatted

def optim_init(optimizer):
    optimizer.state.clear()

def efficiency_loss(alpha):
    first_depth = alpha[:, 0].argmax().item()
    alpha_softmax = F.softmax(alpha, dim=1)
    coef = max(first_depth - alpha.size(0)/2, 0) / (alpha.size(0)/2)
    return coef * alpha_softmax[:first_depth, 1].sum()

def decayed_loss_with_alpha(idx, losses, alpha_softmax):
    if idx == len(losses)-1:
        return losses[-1]

    loss = alpha_softmax[idx, 0] * losses[idx]
    for i in range(idx+1, len(losses) - 1):
        loss += alpha_softmax[idx:i, 1].prod() * alpha_softmax[i, 0] * losses[i]

    loss += alpha_softmax[idx:, 1].prod() * losses[-1]

    return loss

def move_optim_state_dict(state_dict, device):
    sd = deepcopy(state_dict)
    for state in sd['state'].values():
        for k, v in state.items():
            if isinstance(v, torch.Tensor):
                state[k] = v.data.to(device)

    return sd


def weights_init(m):
    if isinstance(m, nn.BatchNorm1d) or isinstance(m, nn.BatchNorm2d):
        nn.init.constant_(m.weight, 1)
        nn.init.constant_(m.bias, 0)
    if isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear):
        a = 0 if config.nonlin == 'relu' else 0.01
        nn.init.kaiming_normal_(m.weight, a=a, nonlinearity='leaky_relu')
        if m.bias is not None:
            nn.init.constant_(m.bias, 0)


def get_singular_values(module):
    s = []
    for m in module.modules():
        if isinstance(m, nn.Conv2d):
            out_channels = m.weight.size(0)
            w = m.weight.data.cpu().numpy().reshape(out_channels, -1)
            s.append(svd(w, compute_uv=False))
        elif isinstance(m, nn.Linear):
            w = m.weight.data.cpu().numpy()
            s.append(svd(w, compute_uv=False))
    return s

def get_correct_sample_indices(forward_fn, x, y):
    with torch.no_grad():
        out = forward_fn(x)
        if isinstance(out, tuple):
            out = out[0]
        pred = torch.max(out, 1)[1]

    correct_idx = (pred == y).nonzero().view(-1)
    if correct_idx.numel() == 0:
        return None
    return correct_idx


def one_hot(y, n_dims):
    scatter_dim = len(y.size())
    y_tensor = y.view(*y.size(), -1)
    zeros = torch.zeros(*y.size(), n_dims).to(y.device)
    return zeros.scatter(scatter_dim, y_tensor, 1)

def similarity_matrix(x):
    ''' Calculate adjusted cosine similarity matrix of size x.size(0) x x.size(0). '''
    if x.dim() == 4:
        if not config.no_similarity_std and x.size(1) > 3 and x.size(2) > 1:
            z = x.view(x.size(0), x.size(1), -1)
            x = z.std(dim=2)
        else:
            x = x.view(x.size(0),-1)
    xc = x - x.mean(dim=1).unsqueeze(1)
    xn = xc / (1e-8 + torch.sqrt(torch.sum(xc**2, dim=1))).unsqueeze(1)
    R = xn.matmul(xn.transpose(1,0)).clamp(-1,1)
    return R



def get_pool_layer(channels, dim, target_dim):
    '''Resolve average-pooling kernel size in order for flattened dim to match target_dim'''
    ks_h, ks_w = 1, 1
    dim_out_h, dim_out_w = dim, dim
    dim_in_decoder = channels*dim_out_h*dim_out_w
    while dim_in_decoder > target_dim and ks_h < dim:
        ks_h*=2
        dim_out_h = math.ceil(dim / ks_h)
        dim_in_decoder = channels*dim_out_h*dim_out_w
        if dim_in_decoder > target_dim:
            ks_w*=2
            dim_out_w = math.ceil(dim / ks_w)
            dim_in_decoder = channels*dim_out_h*dim_out_w
    if ks_h > 1 or ks_w > 1:
        pad_h = (ks_h * (dim_out_h - dim // ks_h)) // 2
        pad_w = (ks_w * (dim_out_w - dim // ks_w)) // 2
        return nn.AvgPool2d((ks_h, ks_w), padding=(pad_h, pad_w)), dim_in_decoder
    else:
        return nn.Identity(), dim_in_decoder

def save_checkpoint(state, num_blocks, is_best):
    """
    Save a copy of the model so that it can be loaded at a future
    date. This function is used when the model is being evaluated
    on the test data.
    If this model has reached the best validation accuracy thus
    far, a seperate file with the suffix `best` is created.
    """
    base_dir = os.path.join(config.ckpt_dir, config.dataset)

    if not os.path.exists(base_dir):
        os.makedirs(base_dir)

    filename = config.name + '_{}_ckpt.pth.tar'.format(num_blocks)
    ckpt_path = os.path.join(base_dir, filename)
    print("[*] Saving model to {}".format(ckpt_path))
    torch.save(state, ckpt_path)

    if is_best:
        filename = config.name + '_{}_model_best.pth.tar'.format(num_blocks)
        shutil.copyfile(
            ckpt_path, os.path.join(base_dir, filename)
        )

def load_checkpoint(model, device, num_blocks, is_best=False):
    """
    Load the best copy of a model. This is useful for 2 cases:
    - Resuming training with the most recent model checkpoint.
    - Loading the best validation model to evaluate on the test data.
    Params
    ------
    - best: if set to True, loads the best model. Use this if you want
        to evaluate your model on the test data. Else, set to False in
        which case the most recent version of the checkpoint is used.
    """
    base_dir = os.path.join(config.ckpt_dir, config.dataset)
    print("[*] Loading model from {}".format(base_dir))

    filename = config.name + '_{}_ckpt.pth.tar'.format(num_blocks)
    if is_best:
        filename = config.name + '_{}_model_best.pth.tar'.format(num_blocks)
    ckpt_path = os.path.join(base_dir, filename)
    ckpt = torch.load(ckpt_path)
    model.load_state_dict(move_model_state_dict(ckpt, device))

    print(
        "[*] Loaded checkpoint {}".format(
            filename)
    )

    return

class CosineAnnealingLR(_LRScheduler):
    r"""Clipped cosine annealing lr. Return eta_min when last_epoch >= T_max"""

    def __init__(self, optimizer, T_max, eta_min=0, last_epoch=-1):
        self.T_max = T_max
        self.eta_min = eta_min
        super(CosineAnnealingLR, self).__init__(optimizer, last_epoch)

    def get_lr(self):
        if self.last_epoch == 0:
            return self.base_lrs
        elif self.last_epoch >= self.T_max:
            return [self.eta_min for _ in range(len(self.optimizer.param_groups))]
        elif (self.last_epoch - 1 - self.T_max) % (2 * self.T_max) == 0:
            return [group['lr'] + (base_lr - self.eta_min) *
                    (1 - math.cos(math.pi / self.T_max)) / 2
                    for base_lr, group in
                    zip(self.base_lrs, self.optimizer.param_groups)]
        return [(1 + math.cos(math.pi * self.last_epoch / self.T_max)) /
                (1 + math.cos(math.pi * (self.last_epoch - 1) / self.T_max)) *
                (group['lr'] - self.eta_min) + self.eta_min
                for group in self.optimizer.param_groups]

    def _get_closed_form_lr(self):
        if self.last_epoch >= self.T_max:
            return [self.eta_min for _ in range(len(self.optimizer.param_groups))]
        else:
            return [self.eta_min + (base_lr - self.eta_min) *
                    (1 + math.cos(math.pi * self.last_epoch / self.T_max)) / 2
                    for base_lr in self.base_lrs]

