import torch
import torch.nn as nn
import torch.nn.functional as F

from config import config
from utils import weights_init

from module import *
from decoder import *
from loss import *
from .convnet import BaseConvNet

cfgs = {
    'vgg11':  [ 16, 'M', 32, 'M', 64, 64, 'M', 128, 128, 'M', 128, 128, 'M'],
    'vgg13':  [ 16,  16, 'M', 32, 32, 'M', 64, 64, 'M', 128, 128, 'M', 128, 128, 'M'],
    'vgg16':  [ 16,  16, 'M', 32, 32, 'M', 64, 64, 64, 'M', 128, 128, 128, 'M', 128, 128, 128, 'M'],
    'vgg19':  [ 16,  16, 'M', 32, 32, 'M', 64, 64, 64, 64, 'M', 128, 128, 128, 128, 'M', 128, 128, 128, 128, 'M'],
}


class VGGn(BaseConvNet):
    '''
    VGG and VGG-like networks.
    The network can be trained by backprop or by locally generated error signal based on cross-entropy and/or similarity matching loss.

    config:
        vgg_name (str): The name of the network.
        feat_mult (float): Multiply number of feature maps with this number.
    '''
    def __init__(self, cfg, **argv):
        super(VGGn, self).__init__(**argv)
        self.cfg = cfg
        feat_mult = config.feat_mult
        output_dim = self._make_layers(self.cfg, self.channels, self.size, feat_mult)

        self.apply(weights_init)

    def _make_layers(self, cfg, channels, size, feat_mult):
        layers = []
        first_layer = True
        scale_cum = 1
        for i, x in enumerate(cfg):
            if x == 'M':
                layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
                scale_cum *=2
            elif x == 'M3':
                layers += [nn.MaxPool2d(kernel_size=3, stride=2, padding=1)]
                scale_cum *=2
            elif x == 'M4':
                layers += [nn.MaxPool2d(kernel_size=4, stride=4)]
                scale_cum *=4
            elif x == 'A':
                layers += [nn.AvgPool2d(kernel_size=2, stride=2)]
                scale_cum *=2
            elif x == 'A3':
                layers += [nn.AvgPool2d(kernel_size=3, stride=2, padding=1)]
                scale_cum *=2
            elif x == 'A4':
                layers += [nn.AvgPool2d(kernel_size=4, stride=4)]
                scale_cum *=4
            else:
                x = int(x * feat_mult)
                layers += [LocalLossBlock(ConvBNAct(channels, x, kernel_size=3, stride=1, padding=1),
                                          self.DecoderClass(x, self.size//scale_cum, self.classes),
                                          self.loss,
                                          'block{}'.format(i)
                                         )]
                channels = x
                first_layer = False
                idx_final_block = i

        self.moduleList.extend(layers[:idx_final_block+1])
        out_dim = size // scale_cum
        decoder = layers[idx_final_block+1:]
        decoder = nn.Sequential(
            *decoder,
            View((-1, out_dim**2*channels)),
            nn.Linear(out_dim**2*channels, 1024*feat_mult),
            nn.ReLU(True),
            nn.Dropout(),
            nn.Linear(1024*feat_mult, 1024*feat_mult),
            nn.ReLU(True),
            nn.Dropout(),
            nn.Linear(1024*feat_mult, self.classes),
        )
        self.moduleList[idx_final_block].decoder = decoder
        return out_dim

def vgg11(**argv):
    return VGGn(cfgs['vgg11'], **argv)

def vgg13(**argv):
    return VGGn(cfgs['vgg13'], **argv)

def vgg16(**argv):
    return VGGn(cfgs['vgg16'], **argv)

def vgg19(**argv):
    return VGGn(cfgs['vgg19'], **argv)
