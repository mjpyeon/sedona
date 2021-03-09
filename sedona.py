'''Meta learning agent'''
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.optim.lr_scheduler import MultiStepLR
import torch.autograd as ag
import higher
import time
from tqdm import tqdm
import os
import shutil
import random

from weight_memory import WeightMemory
from models import get_model
from init_memory import init_memory
from decoder import get_baseline_decoder, MixedPredConvLightDecoder
from loss import SmoothCrossEntropyLoss
from utils import move_model_state_dict, move_optim_state_dict, weights_init, get_formatted_string, AverageMeter, get_error, get_weight_config_str, CosineAnnealingLR
from train import test, continuous_train_step
from config import config
from dataset import DATASET_CONFIGS

EPS = 1e-5

class SEDONA:
    def __init__(self, memory_size=50, max_step=50000):
        self.memory = WeightMemory(memory_size)
        self.max_step = max_step
        self.buffer_scalar = dict()

    def init_models(self, arch, train_loader, valid_loader):
        '''Initialize models, devices, optimizers, and schedulers
        and update them to the memory'''
        decoder = get_baseline_decoder(config.baseline_dec_type) if config.no_beta else MixedPredConvLightDecoder

        self.devices = [torch.device('cuda:{}'.format(i))
                        for i in range(torch.cuda.device_count())]
        model = get_model(arch, 1, decoder=decoder, cuda=False, return_devices=False)[0]
        self.base_device = self.devices[0]
        self.model = model.to(self.base_device)

        self.init_optimizers()

        self.time_step = -1
        self.error = 0

        weight_config_str = get_weight_config_str()
        if not config.not_load_memory and not self.memory.load(
                                     os.path.join(config.mem_dir, arch, weight_config_str)):
            path = os.path.join(config.mem_dir, arch, weight_config_str)
            init_memory(self.memory, self.model, self.base_device, train_loader,
                        valid_loader, self.max_step)
            self.memory.save(path)

    def init_optimizers(self):
        params = [{'params': self.model.block_parameters(j)} for j in range(self.model.num_blocks)]
        if config.optim == 'adam':
            self.optimizer = optim.Adam(params, lr=config.lr, weight_decay=config.weight_decay)
        elif config.optim == 'momentum':
            self.optimizer = optim.SGD(params, lr=config.lr, weight_decay=config.weight_decay, momentum=config.momentum)
        elif config.optim == 'sgd':
            self.optimizer = optim.SGD(params, lr=config.lr, weight_decay=config.weight_decay, momentum=config.momentum)
        else:
            raise NotImplementedError

        if config.lr_scheduler == 'cosine':
            print("Using cosine annealing lr")
            self.scheduler = CosineAnnealingLR(self.optimizer, self.max_step, config.lr_min)
        elif config.lr_scheduler == 'multistep':
            milestones = [config.train_iters // 3 * i for i in range(1, 3)]
            print("Using multistep lr with milestones at {}".format(milestones))
            self.scheduler = MultiStepLR(self.optimizer, milestones)

    def init_variables(self):
        # initialize decision variables
        # for gradient isolation and loss selection
        self.alpha = torch.zeros((self.model.num_blocks-1, 2),
                                    requires_grad=True,
                                    device=self.base_device)
        if config.no_beta:
            self.beta = torch.tensor([[]]*(self.model.num_blocks-1), requires_grad=True, device=self.base_device)
        else:
            self.beta = torch.zeros((self.model.num_blocks-1, self.model.num_decoders),
                                    requires_grad=True,
                                    device=self.base_device)

    def parameters(self):
        return self.alpha, self.beta

    def save_to_memory(self, p=1., indices=None, eviction='oldest'):
        if p < 1. and random.random() > p:
            return

        model_state = move_model_state_dict(self.model.state_dict(), 'cpu')
        optim_state = move_optim_state_dict(self.optimizer.state_dict(), 'cpu')
        step = self.time_step
        error = self.error
        self.memory.update(step, model_state, optim_state, error, eviction=eviction)

    def load_from_memory(self):
        try:
            steps, model_states, optim_states, errors = self.memory.sample(1)
        except Exception as err:
            self.save_to_memory()
            steps, model_states, optim_states, errors = self.memory.sample(1)

        self.time_step = steps[0]
        self.error = errors[0]
        self.model.load_state_dict(move_model_state_dict(model_states[0], self.base_device))
        self.optimizer.load_state_dict(move_optim_state_dict(optim_states[0], self.base_device))

    def diff_step(self, train_loader, valid_loader, num_inner_steps=5, num_valid_batches=1):
        counter_inner = 0
        counter_outer = 0
        valid_error = AverageMeter()
        total_valid_loss = AverageMeter()
        num_blocks = self.model.num_blocks
        loss_fn = SmoothCrossEntropyLoss(config.smoothing)

        alpha_softmax = F.softmax(self.alpha, dim=1)
        dec_out_scale = [t.item() for t in 1/(F.softmax(self.alpha.data, dim=-1)[:, 0]+EPS)] + [1.]
        beta_softmax = F.softmax(self.beta, dim=1)


        lrs = [None]*num_blocks

        device = self.base_device
        self.model.train()
        self.model.zero_grad()
        self.scheduler.step(self.time_step)
        lr = self.optimizer.param_groups[0]['lr']
        with higher.innerloop_ctx(self.model, self.optimizer,
                                  copy_initial_weights=True,
                                  track_higher_grads=True) as (fnet, diffopt):
            exit_inner_loop = False
            # inner loop
            while True:
                for x, y in train_loader:
                    x, y = x.to(device), y.to(device)
                    fnet(x)
                    losses = fnet.get_loss(y, weights=beta_softmax,
                                           categorical=False,
                                           scale=dec_out_scale)
                    loss = losses[-1]
                    for i in range(len(losses)-2, -1, -1):
                        loss = alpha_softmax[i, 1] * loss + alpha_softmax[i, 0] * losses[i]

                    device = self.devices[(counter_inner+1) % len(self.devices)]
                    #deviders = [alpha_softmax.data[:j, 1].prod().item() for j in range(num_blocks)]
                    deviders = [alpha_softmax[:j, 1].prod().to(device) for j in range(num_blocks)]
                    block_grad_callbacks = [(lambda grads, d=d: [(g/d if g is not None else g) for g in grads]) for d in deviders]
                    diffopt.step(loss,
                                 #grad_callback=grad_callback,
                                 grouped_grad_callbacks=block_grad_callbacks,
                                 device=device)



                    counter_inner += 1
                    if counter_inner >= num_inner_steps:
                        exit_inner_loop = True
                        break
                    else:
                        alpha_softmax = alpha_softmax.to(device)
                        beta_softmax = beta_softmax.to(device)

                if exit_inner_loop:
                    break

            # outer loop
            fnet.eval()
            for x, y in valid_loader:
                x, y = x.to(device), y.to(device)
                out = fnet(x, save_act=False)

                top1_val_err = get_error(out, y, (1,))[0].item()
                valid_error.update(top1_val_err, x.size(0))
                total_valid_loss.update(loss_fn(out, y).to(self.base_device), out.size(0))

                counter_outer += 1
                if num_valid_batches > 0 and counter_outer >= num_valid_batches:
                    break

            valid_loss = total_valid_loss.avg
            self.buffer_scalar['meta_loss'] = valid_loss.item()
            self.buffer_scalar['time_step'] = self.time_step
            self.buffer_scalar['lr'] = lr
            self.alpha.grad, self.beta.grad = ag.grad(loss, [self.alpha, self.beta],
                                                allow_unused=True)

        self.error = valid_error.avg
        return valid_loss.item()

    def train_step(self, train_loader, valid_loader=None):
        for x, y in train_loader:
            x, y = x.to(self.base_device), y.to(self.base_device)
            break

        continuous_train_step(x, y, self.time_step, self.model, self.optimizer, self.alpha.data,
                              self.beta.data, categorical=False, device=self.base_device)
        if valid_loader is not None:
            valid_loss, valid_error, _ = test(self.model, valid_loader, device=self.base_device)
            self.buffer_scalar['valid_loss'] = valid_loss
            self.buffer_scalar['valid_error'] = valid_error

    def monitor(self, idx, valid_loader):
        dconfig = DATASET_CONFIGS[config.dataset]
        x = torch.randn(1, dconfig['channels'], dconfig['size'], dconfig['size']).to(self.base_device)

        if not config.not_load_memory:
            memory_statistics = {
                'num_elements': self.memory.num_elements,
                'avg_error': sum(self.memory.get_attr('error')) / self.memory.num_elements
            }

        msg = '\n========================= Step {} =========================\n'.format(idx)
        msg += "[Meta Statistics]\n"
        for k, v in self.buffer_scalar.items():
            msg += '\t{}: {}\n'.format(k, get_formatted_string(v))

        if not config.not_load_memory:
            msg += "[memory Statistics]\n"
            for k, v in memory_statistics.items():
                msg += '\t{}: {}\n'.format(k, get_formatted_string(v))

        return msg

    def save_states(self, base_dir, i, optim_states, random_states):
        '''base_dir/
                mem/ : dump memory
                var/ : dump variables
                    alpha.pt
                    beta.pt
                states.pt: save current iter and random & optim states
        '''
        if not os.path.exists(base_dir):
            os.makedirs(base_dir)
        self.save_to_memory()
        self.memory.save(os.path.join(base_dir, 'mem'))
        self.save_variables(os.path.join(base_dir, 'var'))
        states = {
            'iter': i,
            'optim_states': optim_states, 'random_states': random_states
        }
        torch.save(states, os.path.join(base_dir, 'states.pt'))

    def load_states(self, base_dir, optimizer, remove_after=True):
        self.memory.load(os.path.join(base_dir, 'mem'))
        self.load_from_memory()
        self.load_variables(os.path.join(base_dir, 'var'))
        sd = torch.load(os.path.join(base_dir, 'states.pt'))
        optimizer.load_state_dict(sd['optim_states'])
        torch.random.set_rng_state(sd['random_states'])
        if remove_after:
            shutil.rmtree(base_dir)

    def save_variables(self, base_dir):
        if not os.path.exists(base_dir):
            os.makedirs(base_dir)

        path = os.path.join(base_dir, "alpha.pt")
        torch.save(self.alpha.data.cpu(), path)
        path = os.path.join(base_dir, "beta.pt")
        torch.save(self.beta.data.cpu(), path)

    def load_variables(self, base_dir):
        path = os.path.join(base_dir, "alpha.pt")
        self.alpha.copy_(torch.load(path).to(self.devices[0]))
        if not config.no_beta:
            path = os.path.join(base_dir, "beta.pt")
            self.beta.copy_(torch.load(path).to(self.devices[0]))


