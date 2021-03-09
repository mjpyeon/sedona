from tqdm import tqdm
import random
import os
import threading

import torch
import numpy as np
import torch.optim as optim

from config import config
from dataset import get_dataset, get_dataloader
from utils import weights_init, move_model_state_dict, move_optim_state_dict, get_formatted_string, get_weight_config_str, CosineAnnealingLR
from train import test, continuous_train_step
from models import get_model
from weight_memory import WeightMemory

def save_to_memory(memory, model, optimizer, time_step, error,
                   p=1., eviction='oldest'):
    if p < 1. and random.random() > p:
        return

    model_state = move_model_state_dict(model.state_dict(), 'cpu')
    optim_state = move_optim_state_dict(optimizer.state_dict(), 'cpu')
    steps = time_step
    errors = error
    memory.update(steps, model_state, optim_state, errors, eviction=eviction)

def init_memory(memory, model, device, train_loader, valid_loader,
                train_iters, num_valid_batches=1, init_weights=False):
    '''Prepare weights in the memory across time steps'''
    if init_weights:
        model.apply(weights_init)

    num_blocks = model.num_blocks

    params = [{'params': model.block_parameters(j)} for j in range(model.num_blocks)]
    if config.optim == 'adam':
        optimizer = optim.Adam(params, lr=config.lr, weight_decay=config.weight_decay)
    elif config.optim == 'momentum':
        optimizer = optim.SGD(params, lr=config.lr, weight_decay=config.weight_decay, momentum=config.momentum)
    elif config.optim == 'sgd':
        optimizer = optim.SGD(params, lr=config.lr, weight_decay=config.weight_decay, momentum=config.momentum)
    else:
        raise NotImplementedError

    scheduler = CosineAnnealingLR(optimizer, config.train_iters, config.lr_min)

    time_step = 0
    valid_freq = max(1, train_iters//500)

    print("Initialize memory")
    pbar = tqdm(total=train_iters)

    alpha = torch.zeros((num_blocks-1, 2), device=device)
    while True:
        for x, y in train_loader:
            scheduler.step(time_step)

            model.train()
            x, y = x.to(device), y.to(device)
            continuous_train_step(x, y, time_step, model, optimizer, alpha, None,
                                    False, scheduler, device)

            #if time_steps[0] in save_iters:
            if time_step % valid_freq == 0:
                loss_avg, err_avg, _ = test(model, valid_loader, device, num_valid_batches)
                error = err_avg
                model.train()
                pbar.set_description('[Step {}] loss: {:.2f}, error: {:.2f}'.format(
                    time_step, loss_avg, err_avg), True)

                save_to_memory(memory, model, optimizer, time_step, error, 1., eviction='worst')

            time_step += 1

            pbar.update(1)
            if time_step >= train_iters:
                pbar.close()
                return

def main():
    train_dataset, valid_dataset = get_dataset(config.dataset, train=True,
                                               download=True, return_valid=True)
    train_loader = get_dataloader(train_dataset, config.batch_size, shuffle=True,
                                  classes_per_batch=config.classes_per_batch)
    valid_loader = get_dataloader(valid_dataset, config.valid_batch_size)

    arch = config.name.split('-')[0]
    model, device = get_model(arch, 1, cuda=True, return_devices=True) # should be nn.Sequential
    model, device = model[0], device[0]

    memory = WeightMemory(config.memory_size)

    weight_config_str = get_weight_config_str()
    init_memory(memory, model, device,
                train_loader, valid_loader, config.train_iters)
    memory.save(os.path.join(config.mem_dir, arch, weight_config_str))

if __name__ == "__main__":
    os.environ['PYTHONHASHSEED']=str(config.seed)
    torch.manual_seed(config.seed)
    np.random.seed(config.seed)
    random.seed(config.seed)
    main()
