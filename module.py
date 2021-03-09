import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.autograd as ag
from config import config


class LocalLossBlock(nn.Module):
    '''Abstract class for calculating local loss
    config:
        module (nn.Module): Module to be wrapped
    '''
    def __init__(self, encoder, decoder, loss_fn, name='block'):
        super(LocalLossBlock, self).__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.loss_fn = loss_fn
        self.name = name
        self.act = None

    def forward(self, x, save_act=True):
        h = self.encoder(x)
        if save_act:
            self.act = h

        return h

    def get_loss(self, y, weights=None, categorical=False, scale=1.):
        if scale == 1. and not weights:
            out = self.decoder(self.act)
        else:
            out = scale * self.decoder(self.act/scale) if weights is None else scale * self.decoder(self.act/scale, weights, categorical=categorical)
        self.loss = self.loss_fn(out, y)
        del self.act
        self.act = None
        return self.loss


class View(nn.Module):
    def __init__(self, shape):
        super(View, self).__init__()
        self.shape = shape

    def forward(self, x):
        return x.view(*self.shape)


class ConvBNAct(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False):
        super(ConvBNAct, self).__init__()
        nonlin = nn.ReLU(True) if config.nonlin == 'relu' else nn.LeakyReLU(0.01, inplace=True)
        self.layers = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, stride=stride, padding=padding, bias=bias),
            nn.BatchNorm2d(out_channels),
            nonlin,
        )

    def forward(self, x):
        return self.layers(x)

class SepConvBNAct(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False):
        super(SepConvBNAct, self).__init__()
        nonlin = nn.ReLU(True) if config.nonlin == 'relu' else nn.LeakyReLU(0.01, inplace=True)
        self.layers = nn.Sequential(
            # pw
            nn.Conv2d(in_channels, out_channels, kernel_size=1,
                      stride=1, padding=0, bias=bias),
            nn.BatchNorm2d(out_channels),
            nonlin,
            # dw
            nn.Conv2d(out_channels, out_channels, kernel_size=kernel_size,
                      stride=stride, padding=padding, groups=out_channels,
                      bias=bias),
            nn.BatchNorm2d(out_channels),
            nonlin,
        )

    def forward(self, x):
        return self.layers(x)

class LinearBNAct(nn.Module):
    def __init__(self, in_features, out_features, bias=True):
        super(LinearBNAct, self).__init__()
        nonlin = nn.ReLU(True) if config.nonlin == 'relu' else nn.LeakyReLU(0.01, inplace=True)
        self.layers = nn.Sequential(
            nn.Linear(in_features, out_features, bias=bias),
            nn.BatchNorm1d(out_features),
            nonlin,
        )

    def forward(self, x):
        return self.layers(x)

class InvertedResidual(nn.Module):
    def __init__(self, inp, oup, stride, expand_ratio):
        super(InvertedResidual, self).__init__()
        self.stride = stride
        assert stride in [1, 2]

        hidden_dim = int(round(inp * expand_ratio))
        self.use_res_connect = self.stride == 1 and inp == oup

        layers = []
        if expand_ratio != 1:
            layers.extend([
                nn.Conv2d(inp, hidden_dim, kernel_size=1,
                        stride=1, padding=0, bias=False),
                nn.BatchNorm2d(hidden_dim),
                nn.ReLU(True)])

        layers.extend([nn.Conv2d(hidden_dim, hidden_dim,
                                 kernel_size=3,
                                 stride=stride, padding=1,
                                 groups=hidden_dim, bias=False),
            nn.BatchNorm2d(hidden_dim),
            nn.ReLU(True)])

        # pw-linear
        layers.append(nn.Conv2d(hidden_dim, oup, 1, 1, 0, bias=False)),
        layers.append(nn.BatchNorm2d(oup))
        self.conv = nn.Sequential(*layers)

    def forward(self, x):
        if self.use_res_connect:
            return x + self.conv(x)
        else:
            return self.conv(x)
