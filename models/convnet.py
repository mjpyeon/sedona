import torch
import torch.nn as nn
import torch.nn.functional as F

from config import config
from utils import weights_init

from module import *
from decoder import *
from loss import *
from dataset import DATASET_CONFIGS

from .base import BaseNet

class BaseConvNet(BaseNet):
    def __init__(self, decoder=None):
        super(BaseConvNet, self).__init__()
        self.planes = config.planes

        if decoder is None:
            self.DecoderClass = MixedPredConvLightDecoder
        else:
            self.DecoderClass = decoder

        if decoder == PredSimConvDecoder:
            #self.loss = PredSimLoss(0.99, self.classes, config.smoothing)
            self.loss = PredSimLoss(0.99, self.classes)
        else:
            self.loss = SmoothCrossEntropyLoss(config.smoothing)
