import os
import sys
import pickle
from pathlib import Path

from PIL import Image
import numpy as np
import six


import torch
from torchvision import datasets, transforms
from torch.utils.data import Dataset, DataLoader, Subset
from torch.utils.data.dataloader import default_collate


from config import config

kwargs = {'num_workers': config.num_workers, 'pin_memory': True if 'imagenet' in config.dataset else False}


class NClassRandomSampler(torch.utils.data.sampler.Sampler):
    r'''Samples elements such that most batches have N classes per batch.
    Elements are shuffled before each epoch.
    Arguments:
        targets: target class for each example in the dataset
        n_classes_per_batch: the number of classes we want to have per batch
    '''

    def __init__(self, targets, n_classes_per_batch, batch_size):
        self.targets = targets
        self.n_classes = int(np.max(targets))
        self.n_classes_per_batch = n_classes_per_batch
        self.batch_size = batch_size

    def __iter__(self):
        n = self.n_classes_per_batch

        ts = list(self.targets)
        ts_i = list(range(len(self.targets)))

        np.random.shuffle(ts_i)
        #algorithm outline:
        #1) put n examples in batch
        #2) fill rest of batch with examples whose class is already in the batch
        while len(ts_i) > 0:
            idxs, ts_i = ts_i[:n], ts_i[n:] #pop n off the list

            t_slice_set = set([ts[i] for i in idxs])

            #fill up idxs until we have n different classes in it. this should be quick.
            k = 0
            while len(t_slice_set) < 10 and k < n*10 and k < len(ts_i):
                if ts[ts_i[k]] not in t_slice_set:
                    idxs.append(ts_i.pop(k))
                    t_slice_set = set([ts[i] for i in idxs])
                else:
                    k += 1

            #fill up idxs with indexes whose classes are in t_slice_set.
            j = 0
            while j < len(ts_i) and len(idxs) < self.batch_size:
                if ts[ts_i[j]] in t_slice_set:
                    idxs.append(ts_i.pop(j)) #pop is O(n), can we do better?
                else:
                    j += 1

            if len(idxs) < self.batch_size:
                needed = self.batch_size-len(idxs)
                idxs += ts_i[:needed]
                ts_i = ts_i[needed:]

            for i in idxs:
                yield i

    def __len__(self):
        return len(self.targets)


def get_train_valid_split(dataset):
    np.random.seed(config.seed)

    num_train = len(dataset)
    indices = list(range(num_train))
    split_idx = int(np.floor(config.valid_size * num_train))

    np.random.seed(config.seed)
    np.random.shuffle(indices)

    train_idx = indices[split_idx:]
    valid_idx = indices[:split_idx]
    indices = {'train': train_idx, 'valid': valid_idx}

    return indices



def get_dataset(name, train=True, download=True, return_valid=False):

    dataset_class = AVAILABLE_DATASETS[name]
    test_transform = AVAILABLE_TRANSFORMS[False][name]
    dataset_transform = AVAILABLE_TRANSFORMS[train][name]

    dataset_transform = transforms.Compose([
        *dataset_transform
    ])
    test_transform = transforms.Compose([
        *test_transform
    ])


    if 'cifar' in name:
        dataset = dataset_class('./{data}/{name}'.format(data=config.data_dir, name=name), train=train,
                                    download=download, transform=dataset_transform,
                                    )
        if train and return_valid:
            valid_dataset = dataset_class('./{data}/{name}'.format(data=config.data_dir, name=name), train=train,
                                            download=download, transform=test_transform,
                                            )
            indices = get_train_valid_split(dataset)
            return Subset(dataset, indices['train']), Subset(valid_dataset, indices['valid'])
        else:
            return dataset

    elif name == 'imagenet':
        _split = 'train' if train else 'val'
        dataset_dir = './{data}/{name}/{split}'.format(data=config.data_dir, name=name, split=_split)

        dataset = dataset_class(dataset_dir,
            transform=dataset_transform)
        if train and return_valid:
            valid_dataset = dataset_class('./{data}/{name}/{split}'.format(data=config.data_dir, name=name, split='val'),
                                            transform=test_transform)
            return dataset, valid_dataset
        else:
            return dataset

    elif name == 'tiny-imagenet':
        _split = 'train' if train else 'test'
        dataset_dir = './{data}/{name}/{split}'.format(data=config.data_dir, name=name, split=_split)

        dataset = dataset_class(dataset_dir,
            transform=dataset_transform)
        if train and return_valid:
            valid_dataset = dataset_class('./{data}/{name}/{split}'.format(data=config.data_dir, name=name, split='val'),
                                            transform=test_transform)
            return dataset, valid_dataset
        else:
            return dataset

    else:
        raise NotImplementedError



def get_dataloader(dataset, batch_size=128, shuffle=False, classes_per_batch=0):
    size, channels, classes = DATASET_CONFIGS[config.dataset]['size'], DATASET_CONFIGS[config.dataset]['channels'], DATASET_CONFIGS[config.dataset]['classes']

    shuffle = False if classes_per_batch > 0 else shuffle

    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle = shuffle,
        sampler = None if classes_per_batch == 0 else NClassRandomSampler(np.array(dataset.targets), classes_per_batch, batch_size),
        **kwargs
    )


AVAILABLE_DATASETS = {
    'cifar10': datasets.CIFAR10,
    'tiny-imagenet': datasets.ImageFolder,
    'imagenet': datasets.ImageFolder,
}

AVAILABLE_TRANSFORMS = {
    True:{
        'cifar10': [
            transforms.RandomCrop(32, padding=2),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2470, 0.2435, 0.2616))
        ],
        'tiny-imagenet': [
            transforms.RandomRotation(20),
            transforms.RandomCrop(64, padding=4),
            transforms.RandomHorizontalFlip(0.5),
            transforms.ToTensor(),
            transforms.Normalize([0.4802, 0.4481, 0.3975], [0.2302, 0.2265, 0.2262]),
        ],
        'imagenet': [
            transforms.RandomResizedCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
        ],
    },

    False:{
        'cifar10': [
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2470, 0.2435, 0.2616))
        ],
        'tiny-imagenet': [
            transforms.ToTensor(),
            transforms.Normalize([0.4802, 0.4481, 0.3975], [0.2302, 0.2265, 0.2262]),
        ],
        'imagenet': [
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
        ],
    }
}

DATASET_CONFIGS = {
    'cifar10': {'size': 32, 'channels': 3, 'classes': 10},
    'tiny-imagenet': {'size': 64, 'channels': 3, 'classes': 200},
    'imagenet': {'size': 224, 'channels': 3, 'classes': 1000},
}


