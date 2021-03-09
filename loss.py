import torch
import torch.nn as nn
from torch.nn.modules.loss import _WeightedLoss
import torch.nn.functional as F
from utils import similarity_matrix, one_hot
from config import config

class SimLoss(nn.Module):
    def __init__(self, classes):
        super(SimLoss, self).__init__()
        self.classes = classes

    def forward(self, Sh, y):
        y_onehot = one_hot(y, self.classes)
        Sy = similarity_matrix(y_onehot)
        return F.mse_loss(Sh, Sy)


class PredSimLoss(nn.Module):
    def __init__(self, beta, classes, smoothing=0.1):
        super(PredSimLoss, self).__init__()
        self.beta = beta
        #self.pred = SmoothCrossEntropyLoss(smoothing)
        self.pred = nn.CrossEntropyLoss()
        self.sim = SimLoss(classes)

    def forward(self, outputs, y):
        if isinstance(outputs, tuple):
            logit, Sh = outputs
            loss = (1.-self.beta) * self.pred(logit, y) + self.beta * self.sim(Sh, y)
        else:
            loss = self.pred(outputs, y)
        return loss


class SmoothCrossEntropyLoss(_WeightedLoss):
    def __init__(self, smoothing=0.1, weight=None, reduction='mean'):
        super().__init__(weight=weight, reduction=reduction)
        self.smoothing = smoothing
        self.weight = weight
        self.reduction = reduction

    @staticmethod
    def _smooth_one_hot(targets:torch.Tensor, n_classes:int, smoothing=0.0):
        # assert 0 <= smoothing < 1
        # assert 0 <= targets.min(), "targets.min() is {}, smaller than 0".format(targets.min())
        # assert targets.max() < n_classes, "targets.max() is {}, larger than {}".format(targets.max(), n_classes-1)
        with torch.no_grad():
            targets = torch.empty(size=(targets.size(0), n_classes),
                    device=targets.device) \
                .fill_(smoothing /(n_classes-1)) \
                .scatter_(1, targets.data.unsqueeze(1), 1.-smoothing)
        return targets

    def forward(self, inputs, targets):
        if self.smoothing > 0:
            targets = SmoothCrossEntropyLoss._smooth_one_hot(targets, inputs.size(-1),
                self.smoothing)
            lsm = F.log_softmax(inputs, -1)

            if self.weight is not None:
                lsm = lsm * self.weight.unsqueeze(0)

            loss = -(targets * lsm).sum(-1)

            if  self.reduction == 'sum':
                loss = loss.sum()
            elif  self.reduction == 'mean':
                loss = loss.mean()
        else:
            loss = F.cross_entropy(inputs, targets, weight=self.weight,
                                   reduction=self.reduction)

        return loss
