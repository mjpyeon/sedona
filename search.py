import os
from glob import glob
import json
import random

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.backends import cudnn
import torchvision.utils as vutils
from torchvision import transforms
from tqdm import tqdm

from config import config
from sedona import SEDONA
from dataset import get_dataset, get_dataloader, DATASET_CONFIGS, AVAILABLE_TRANSFORMS
from models import get_model


def search(train_dataset, valid_dataset):
    train_loader = get_dataloader(train_dataset, config.batch_size, shuffle=True,
                                  classes_per_batch=config.classes_per_batch)
    valid_shuffled_loader = get_dataloader(valid_dataset, config.valid_batch_size, shuffle=True,
                                           classes_per_batch=config.classes_per_batch)
    valid_loader = get_dataloader(valid_dataset, config.valid_batch_size)

    meta_learner = SEDONA(config.memory_size, config.train_iters)
    meta_learner.init_models(config.name.split('-')[0], train_loader, valid_loader)
    meta_learner.init_variables()

    if config.meta_optimizer == 'adam':
        meta_optimizer = optim.Adam(meta_learner.parameters(), lr=config.meta_lr, betas=(0.5, 0.999), weight_decay=config.meta_wd)
    elif config.meta_optimizer == 'momentum':
        meta_optimizer = optim.SGD(meta_learner.parameters(), lr=config.meta_lr, momentum=0.9, weight_decay=config.meta_wd)
    elif config.meta_optimizer == 'sgd':
        meta_optimizer = optim.SGD(meta_learner.parameters(), lr=config.meta_lr, weight_decay=config.meta_wd)

    tmp_dir = os.path.join(config.tmp_dir, config.name)
    if config.resume:
        # load the latest state
        dirs = glob(os.path.join(tmp_dir, '*'))
        start_iter = max(int(d.split('/')[-1]) for d in dirs)
        tmp_dir = os.path.join(tmp_dir, str(start_iter))
        meta_learner.load_states(tmp_dir, meta_optimizer, remove_after=True)
    else:
        start_iter = 0

    pbar = tqdm(initial=start_iter, total=config.meta_train_iters)

    for i in range(start_iter, config.meta_train_iters):
        if not config.not_load_memory:
            meta_learner.load_from_memory()
        meta_learner.time_step += 1
        meta_optimizer.zero_grad()
        loss = meta_learner.diff_step(train_loader, valid_shuffled_loader, config.num_inner_steps, 1)
        meta_optimizer.step()

        meta_learner.train_step(train_loader, valid_loader=valid_loader if (i+1) % config.monitor_freq==0 else None)
        if not config.not_load_memory:
            meta_learner.save_to_memory()
        if (i+1) % config.monitor_freq == 0:
            msg = meta_learner.monitor(i, valid_loader)
            pbar.write(msg)

        pbar.update(1)

    pbar.close()

    alpha, beta = meta_learner.parameters()
    alpha, beta = alpha.data.cpu(), beta.data.cpu()
    return alpha, beta

def main():
    # prepare datasets
    train_dataset, valid_dataset = get_dataset(config.dataset, train=True, download=True, return_valid=True)

    # search
    alpha, beta = search(train_dataset, valid_dataset)

    # save alpha and beta
    base_dir = os.path.join(config.out_dir, config.name)
    if not os.path.exists(base_dir):
        os.makedirs(base_dir)

    path = os.path.join(base_dir, "alpha.pt")
    torch.save(alpha, path)
    path = os.path.join(base_dir, "beta.pt")
    torch.save(beta, path)


if __name__ == "__main__":
    os.environ['PYTHONHASHSEED']=str(config.seed)
    torch.manual_seed(config.seed)
    np.random.seed(config.seed)
    random.seed(config.seed)

    main()

