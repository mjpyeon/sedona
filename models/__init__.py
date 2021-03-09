from .convnet import *
from .vgg import *
from .resnet import *

import torch as _torch

def get_model(arch, num_models=1, decoder=None, cuda=False, return_devices=True):
    models = [eval(arch)(decoder=decoder) for _ in range(num_models)]

    if cuda:
        n_gpus = _torch.cuda.device_count()
        devices = [_torch.device('cuda:{}'.format(i % n_gpus)) for i in range(num_models)]
        models = [models[i].to(devices[i]) for i in range(num_models)]
    else:
        devices = [_torch.device('cpu') for _ in range(num_models)]

    if return_devices:
        return models, devices
    else:
        return models

