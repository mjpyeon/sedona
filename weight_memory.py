import torch
import torch.optim as optim
import time
import math
from collections.abc import Iterable
import numpy as np
import json
import os
from glob import glob

from queue import PriorityQueue

from models import get_model
from config import config

class WeightMemory:
    def __init__(self, memory_size):
        self.memory_size = memory_size
        self.mem = dict()
        self.oldest = None

    @property
    def num_elements(self):
        return len(self.mem.keys())

    @property
    def key_oldest(self):
        oldest_t = math.inf
        for k in self.mem.keys():
            if k < oldest_t:
                oldest_t  = k

        if oldest_t == math.inf:
            return None
        else:
            return oldest_t

    @property
    def key_worst(self):
        worst_error = 0.
        worst_key = None
        for k, element in self.mem.items():
            err = element['error']
            if err > worst_error:
                worst_error  = err
                worst_key = k

        return worst_key

    def get_attr(self, name_attr):
        return [e[name_attr] for e in self.mem.values()]

    def update(self, steps, model_states, optim_states, errors, eviction='oldest'):
        if isinstance(steps, Iterable):
            for i, step in enumerate(steps):
                self._update(step, model_states[i], optim_states[i], errors[i])
        else:
            self._update(steps, model_states, optim_states, errors)

    def _update(self, step, model_state_dict, optim_state_dict, error, eviction='oldest'):
        if step in self.mem.keys(): del self.mem[step]
        if self.num_elements == self.memory_size:
            if eviction == 'oldest':
                del self.mem[self.key_oldest]
            elif eviction =='worst':
                del self.mem[self.key_worst]
            else:
                raise NotImplementedError

        self.mem[time.time()] = {
            'step': step,
            'model_state': model_state_dict,
            'optim_state': optim_state_dict,
            'error': error
        }


    def sample(self, k):
        if self.num_elements < k:
            raise Exception(k-self.num_elements)

        times = np.random.choice(list(self.mem.keys()), size=(k,),
                                 replace=False)
        elements = [self.mem.pop(i) for i in times]
        steps = [e['step'] for e in elements]
        model_states = [e['model_state'] for e in elements]
        optim_states = [e['optim_state'] for e in elements]
        errors = [e['error'] for e in elements]
        return steps, model_states, optim_states, errors

    def save(self, save_dir):
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
            print("Overwrite memory images in {}".format(save_dir))

        for k in self.mem.keys():
            path = os.path.join(save_dir, str(k))
            torch.save(self.mem[k], path)

    def load(self, load_dir):
        if not os.path.exists(load_dir):
            print("Undable to load memory image from {}".format(load_dir))
            return False

        fnames = sorted(glob(os.path.join(load_dir, '*')))[-self.memory_size:]
        keys = [float(fname.split('/')[-1]) for fname in fnames]
        for k in keys:
            path = os.path.join(load_dir, str(k))
            loaded = torch.load(path)
            self.mem[k] = loaded
        print("Load {} weights into the memory".format(len(keys)))

        return True

