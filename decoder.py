'''Decoder modules'''

import torch.nn as nn
import torch.nn.functional as F
from config import config
from module import View
from utils import similarity_matrix, get_pool_layer, one_hot
from module import *

from itertools import product
import math


class identity(nn.Module):
    def __init__(self):
        super(identity, self).__init__()

    def forward(self, input):
        return input


class SimilarityMatrix(nn.Module):
    def __init__(self):
        super(SimilarityMatrix, self).__init__()

    def forward(self, x):
        return similarity_matrix(x)


class Decoder(nn.Module):
    def __init__(self):
        super(Decoder, self).__init__()


class PredDecoder(Decoder):
    def __init__(self):
        super(Decoder, self).__init__()

    def forward(self, x):
        return self.layers(x)


class SimDecoder(Decoder):
    def __init__(self):
        super(Decoder, self).__init__()
        self.sim_module = SimilarityMatrix()

    def forward(self, x):
        out = self.layers(x)
        return self.sim_module(out)


class PredSimDecoder(Decoder):
    def __init__(self):
        super(Decoder, self).__init__()

    @property
    def num_layers(self):
        return self.pred.num_layers

    def forward(self, x):
        return (self.pred(x), self.sim(x))


class PredConvDecoder(PredDecoder):
    def __init__(self, channels, dim, classes, bias=True):
        super(PredConvDecoder, self).__init__()
        avg_pool, dim_in_decoder = get_pool_layer(channels, dim, config.feat_mult * config.dim_in_decoder)
        if dim_in_decoder is None:
            dim_in_decoder = config.feat_mult * config.dim_in_decoder
        self.layers = nn.Sequential(
            avg_pool,
            View((-1, dim_in_decoder)),
            nn.Linear(dim_in_decoder, classes, bias)
        )

class SimConvDecoder(SimDecoder):
    def __init__(self, channels, dim=None, classes=None, kernel_size=3, stride=1, padding=1, bias=False):
        super(SimConvDecoder, self).__init__()
        self.layers = nn.Sequential(
            nn.Conv2d(channels, channels, kernel_size=kernel_size, stride=stride, padding=padding, bias=bias)
        )


class PredSimConvDecoder(PredSimDecoder):
    def __init__(self, channels, dim, classes, kernel_size=3, stride=1, padding=1):
        super(PredSimConvDecoder, self).__init__()
        self.pred = PredConvDecoder(channels, dim, classes, bias=True)
        self.sim = SimConvDecoder(channels, dim, classes, kernel_size, stride, padding, bias=False)


class MLPSR(Decoder):
    def __init__(self, channels=256, size=32, classes=10, n_lin=3,
                 mlp_layers=3,  batchn=True, bias=True):
        super(MLPSR, self).__init__()
        self.n_lin=n_lin
        self.size=size

        self.init_pool = nn.AdaptiveAvgPool2d(math.ceil(self.size/4))
        self.blocks = []
        for n in range(self.n_lin):
            if batchn:
                bn_temp = nn.BatchNorm2d(channels)
            else:
                bn_temp = identity()

            conv = nn.Conv2d(channels, channels,
                             kernel_size=1, stride=1, padding=0, bias=False)
            relu = nn.ReLU(True)
            self.blocks.append(nn.Sequential(conv,bn_temp,relu))

        self.blocks = nn.ModuleList(self.blocks)

        self.mlp_in_size = min(math.ceil(self.size/4), 2)
        self.out_pool = nn.AdaptiveAvgPool2d(self.mlp_in_size)
        mlp_feat = channels * (self.mlp_in_size) * (self.mlp_in_size)
        layers = []

        for l in range(mlp_layers):
            if l==0:
                in_feat = channels*self.mlp_in_size**2
                mlp_feat = mlp_feat
            else:
                in_feat = mlp_feat

            if batchn:
                bn_temp = nn.BatchNorm1d(mlp_feat)
            else:
                bn_temp = identity()

            layers +=[nn.Linear(in_feat, mlp_feat, bias=bias),
                            bn_temp, nn.ReLU(True)]
        layers += [nn.Linear(mlp_feat, classes, bias=bias)]
        self.classifier = nn.Sequential(*layers)
        self.num_layers = n_lin + mlp_layers + 1

    def forward(self, x):
        out = x
        out = self.init_pool(out)

        for n in range(self.n_lin):
            out = self.blocks[n](out)

        out = self.out_pool(out)
        out = out.view(out.size(0), -1)
        out = self.classifier(out)
        return out


class PredConvLightDecoder(PredDecoder):
    def __init__(self, channels, size, classes, dim_in_decoder, expand_ratio=4, num_layers=1, final_dim=1024, bias=True):
        super(PredConvLightDecoder, self).__init__()
        self.in_planes = channels
        self.size = curr_size = size
        planes = dim_in_decoder
        layers = []
        channel_increment = planes // 2 if expand_ratio > 1 else planes

        layers.append(SepConvBNAct(self.in_planes, planes, stride=2, bias=False))
        self.in_planes = planes
        curr_size = curr_size // 2

        out_planes = planes
        for i in range(num_layers-1):
            stride = 2 if curr_size >= 2 and i % 2 == 1 else 1
            out_planes = out_planes + channel_increment  if stride == 2 else out_planes

            layers.append(InvertedResidual(self.in_planes, out_planes,
                                           expand_ratio=expand_ratio,
                                           stride=stride))
            self.in_planes = out_planes
            curr_size = curr_size // 2 if stride == 2 else curr_size

        layers.append(ConvBNAct(self.in_planes, final_dim,
                                kernel_size=1, padding=0, bias=False))
        self.in_planes = final_dim

        layers.append(nn.AdaptiveAvgPool2d(1))
        layers.append(View((-1, self.in_planes)))
        layers.append(nn.Linear(self.in_planes, classes, bias))
        self.num_layers = num_layers
        self.layers = nn.Sequential(*layers)


class MixedPredConvLightDecoder(PredDecoder):
    def __init__(self, channels, size, classes, bias=True):
        super(MixedPredConvLightDecoder, self).__init__()
        feat_mult = config.feat_mult
        expand_ratio = 1 if config.name.split('-')[0] == 'resnet18' or config.name.split('-')[0] == 'resnet34' else 4
        self.possible_dim_in_decoders =[channels//expand_ratio]
        self.possible_num_layers = [i+1 for i in range(config.min_dec_depth, config.max_dec_depth)]
        self.decoders = nn.ModuleList()
        self.decoder_configs = []
        self.final_dim = 512*feat_mult

        for dim_in_decoder, num_layer in product(self.possible_dim_in_decoders, self.possible_num_layers):
            self.decoders.append(PredConvLightDecoder(channels, size, classes,
                                                 dim_in_decoder, num_layers=num_layer,
                                                 final_dim=self.final_dim,
                                                 expand_ratio = expand_ratio,
                                                 bias=bias))
            self.decoder_configs.append({
                'dim_in_decoder': dim_in_decoder,
                'num_layers': num_layer
            })

    @property
    def num_layers(self):
        '''Return maximum number of layers of all decoders'''
        num_layers = 0
        for decoder in self.decoders:
            num_layers = max(decoder.num_layers, num_layers)

        return num_layers

    @property
    def num_decoders(self):
        return len(self.decoders)

    def forward(self, x, weights=None, categorical=False):
        if weights is not None:
            if categorical:
                max_weight_idx = weights.max(0)[1].item()
                return self.decoders[max_weight_idx](x)*weights[max_weight_idx]
            else:
                return sum(weights[i] * self.decoders[i](x) for i in range(self.num_decoders))
        else:
            return sum(decoder(x) for decoder in self.decoders) / self.num_decoders


def get_baseline_decoder(name=''):
    if name == 'mlp_sr':
        decoder = MLPSR
    elif name == 'predsim':
        decoder = PredSimConvDecoder
    else:
        raise NotImplementedError

    return decoder

