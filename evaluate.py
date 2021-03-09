import torch.optim as optim
import torch.nn as nn
import torch
from filelock import FileLock
import time
import os

from dataset import get_dataset, get_dataloader, DATASET_CONFIGS, AVAILABLE_TRANSFORMS
from train import train, train_parallel, test, test_ensemble
from models import get_model
from config import config
from decoder import MixedPredConvLightDecoder, get_baseline_decoder
from utils import save_checkpoint, load_checkpoint
from torch.optim.lr_scheduler import CosineAnnealingLR, MultiStepLR


def prepare_alpha_beta(target_num_blocks=1, num_blocks=1, num_decoders=3):
    if config.eval_base_dir:
        # load alpha and beta
        alpha_path = os.path.join(config.eval_base_dir, 'alpha.pt')
        alpha = torch.load(alpha_path)
        if config.no_beta:
            beta = None
        else:
            beta_path = os.path.join(config.eval_base_dir, 'beta.pt')
            beta = torch.load(beta_path)
    else:
        # prepare baseline (uniform split)
        alpha = torch.zeros((num_blocks-1, 2))
        alpha[:, 0] = -1
        selected_layers = torch.linspace(-1, num_blocks-1, target_num_blocks+1)[1:-1].long()
        alpha[selected_layers, 0] = 1.

        beta = None

    return alpha, beta


def evaluate_hps(arch, alpha, beta, train_dataset, test_dataset,
                 target_num_blocks=0, valid_dataset=None,
                 decoder=None):
    decoder = MixedPredConvLightDecoder
    if beta is None:
        decoder = get_baseline_decoder(config.baseline_dec_type)

    n_gpus = torch.cuda.device_count()
    model = get_model(arch, 1, cuda=False, return_devices=False, decoder=decoder)[0]
    if config.eval_continuous:
        block_indices, decoder_configs = [], []
    else:
        block_indices, decoder_configs = model.update_model(
            alpha, beta, num_blocks=target_num_blocks
        )
    device = 'cuda:0'
    model = model.to(device)

    if config.optim == 'adam':
        optimizers = [optim.Adam(model.block_parameters(j), lr=config.lr, weight_decay=config.weight_decay)
                    for j in range(model.num_blocks)]
    elif config.optim == 'sgd':
        optimizers = [optim.SGD(model.block_parameters(j), lr=config.lr, momentum=0., weight_decay=config.weight_decay)
                    for j in range(model.num_blocks)]
    elif config.optim == 'momentum':
        optimizers = [optim.SGD(model.block_parameters(j), lr=config.lr, momentum=config.momentum, weight_decay=config.weight_decay)
                    for j in range(model.num_blocks)]
    else:
        raise NotImplementedError

    if config.lr_scheduler == 'cosine':
        print("Using cosine annealing lr")
        schedulers = [CosineAnnealingLR(optimizers[j], config.train_iters, config.lr_min)
                    for j in range(len(optimizers))]
    elif config.lr_scheduler == 'multistep':
        milestones = config.lr_decay_milestones
        print("Using multistep lr with milestones at {}".format(milestones))
        schedulers = [MultiStepLR(optimizers[j], milestones)
                    for j in range(len(optimizers))]
    else:
        raise NotImplementedError

    print("Train model with found backward path ({} blocks)".format(model.num_blocks))
    dconfig = DATASET_CONFIGS[config.dataset]

    train_loader = get_dataloader(train_dataset, config.batch_size, shuffle=True,
                                  classes_per_batch=config.classes_per_batch)
    if valid_dataset:
        valid_loader = get_dataloader(valid_dataset, config.valid_batch_size)
    else:
        valid_loader = None

    test_loader = get_dataloader(test_dataset, config.valid_batch_size)

    if config.test:
        load_checkpoint(model, device, target_num_blocks, is_best=True)
    elif config.eval_continuous:
        sd = train(model, optimizers, train_loader, config.train_iters, alpha=alpha, beta=beta,
                    scheduler=schedulers, valid_loader=valid_loader, valid_freq=config.valid_freq)
    elif n_gpus == 1 or target_num_blocks == 1:
        sd = train(model, optimizers, train_loader, config.train_iters,
                    scheduler=schedulers, valid_loader=valid_loader, mixed_precision=config.fp16,
                   valid_freq=config.valid_freq)
    elif n_gpus > 0:
        model.to('cpu')
        devices = [torch.device('cuda:{}'.format(i % n_gpus)) for i in range(model.num_blocks)]
        sd = train_parallel(model, optimizers, train_loader, config.train_iters,
                            schedulers, devices, valid_loader=valid_loader,
                            mixed_precision=config.fp16, valid_freq=config.valid_freq)
        model.to(device)
    else:
        raise NotImplementedError

    if not config.test:
        save_checkpoint(sd, target_num_blocks, is_best=True)

    if not config.test:
        test_loss, err1, err5 = test(model, test_loader, device=device, mixed_precision=config.fp16)
    else:
        test_loss, err1, err5 = test_ensemble(model, test_loader, device=device, num_ensemble=config.num_ensemble, mixed_precision=config.fp16)

    print("Test error: {:.2f}% (top1) / {:.2f}% (top5)".format(err1*100, err5*100))

    if not config.test or config.num_ensemble == 1:
        results = {'block_indices': block_indices, 'decoder_configs':decoder_configs,
                'loss': test_loss, 'err@1': err1, 'err@5': err5,
                }
    else:
        results = None

    return results




if __name__ == '__main__':
    import os
    import torch
    import json
    import numpy as np
    import random

    os.environ['PYTHONHASHSEED']=str(config.seed)
    torch.manual_seed(config.seed)
    np.random.seed(config.seed)
    random.seed(config.seed)

    # prepare datasets
    train_dataset, valid_dataset = get_dataset(config.dataset, train=True, download=True, return_valid=True)
    test_dataset = get_dataset(config.dataset, train=False, download=False)

    # evaluate found backward path
    arch = config.name.split('-')[0]
    model = get_model(arch, 1, cuda=False, return_devices=False)[0]
    num_blocks = model.num_blocks
    num_decoders = model.num_decoders
    if config.eval_continuous:
        lst_num_blocks = [num_blocks]
    else:
        lst_num_blocks = range(num_blocks, -1, -1) if config.target_num_blocks == 0 else [config.target_num_blocks]
    for i in lst_num_blocks:
        alpha, beta = prepare_alpha_beta(i, num_blocks, num_decoders)
        results = evaluate_hps(arch, alpha, beta, train_dataset, test_dataset,
                            target_num_blocks=i, valid_dataset=valid_dataset)

        if results is not None:
            result_base_dir = os.path.join(config.result_dir, config.dataset, str(config.feat_mult), config.name)
            result_dir = os.path.join(result_base_dir, str(i)) if not config.eval_continuous else config.result_dir
            if not os.path.exists(result_dir):
                os.makedirs(result_dir)
            result_path = os.path.join(result_dir, 'results.json')
            lock = FileLock(result_path + ".lock")
            with lock:
                with open(result_path, 'w') as f:
                    json.dump(results, f)

            while os.stat(result_path).st_size == 0:
                time.time(1)
            print('write results in {} with size {}'.format(result_path, os.stat(result_path).st_size))


