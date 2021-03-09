import os
import argparse
import torch
from datetime import datetime


parser = argparse.ArgumentParser(description='PyTorch local error training')

# experimental environments
parser.add_argument('--name', help='id of exp ([ARCH_NAME]-[METHOD]-[MISC])')
parser.add_argument('--ckpt-dir', type=str, default='ckpt',
                    help='dir for checkpoints')
parser.add_argument('--out-dir', type=str, default='sedona_outputs',
                    help='dir for saving results')
parser.add_argument('--tmp-dir', type=str, default='temp',
                    help='dir for saving results')
parser.add_argument('--data-dir', type=str, default='data',
                    help='dir containing datasets')
parser.add_argument('--mem-dir', type=str, default='memory_dumps',
                    help='dir for dumping elements in weight memory')
parser.add_argument('--resume', action='store_true', default=False,
                    help='resume or not')
parser.add_argument('--no-cuda', action='store_true', default=False,
                    help='disable CUDA training')
parser.add_argument('--monitor-freq', type=int, default=10,
                    help='frequency of logging')
parser.add_argument('--seed', type=int, default=2021,
                    help='random seed')

# dataset configurations
parser.add_argument('--dataset', default='cifar10',
                    help='cifar10, tiny-imagenet, or ImageNet (default: cifar10)')
parser.add_argument('--valid-size', type=float, default=0.1,
                    help='fraction of valid set')
parser.add_argument('--classes-per-batch', type=int, default=0,
                    help='aim for this number of different classes per batch during training (default: 0, random batches)')
parser.add_argument('--classes-per-batch-until-iter', type=int, default=0,
                    help='limit number of classes per batch until this epoch (default: 0, until end of training)')
parser.add_argument('--num-workers', type=int, default=8,
                    help='number of data loader\'s workers')


# meta learning arguments
parser.add_argument('--num-models', type=int, default=1,
                    help='num of model batches')
parser.add_argument('--meta-train-iters', type=int, default=2000,
                    help='number of batches used to train models')
parser.add_argument('--memory-size', type=int, default=50,
                    help='max num of elements in a weight memory')
parser.add_argument('--num-inner-steps', type=int, default=5,
                    help='num of iters of updating models\' parameters with decision variables')
parser.add_argument('--meta-lr', type=float, default=1e-2,
                    help='initial meta learning rate (default:1e-1)')
parser.add_argument('--meta-optimizer', default='adam',
                    help='optimizer, adam, momentum or sgd (default: adam)')
parser.add_argument('--meta-wd', type=float, default=1e-6,
                    help='weight decay for parameters of meta learner')
parser.add_argument('--min-dec-depth', type=int, default=0,
                    help='minimum depth of decoders')
parser.add_argument('--max-dec-depth', type=int, default=4,
                    help='maximum depth of decoders')
parser.add_argument('--no-beta', action='store_true', default=False,
                    help='not to search decoder architecture')
parser.add_argument('--not-load-memory', default=False, action='store_true',
                    help='whether to load saved weights or not')


# learning arguments
parser.add_argument('--batch-size', type=int, default=128,
                    help='input batch size for training (default: 128)')
parser.add_argument('--valid-batch-size', type=int, default=1024,
                    help='input batch size for validation (default: 1024)')
parser.add_argument('--train-iters', type=int, default=64000,
                    help='number of batches used to train models')
parser.add_argument('--lr', type=float, default=1e-1,
                    help='initial learning rate (default: 5e-4)')
parser.add_argument('--lr-min', type=float, default=1e-3,
                    help='final learning rate (default: 5e-4)')
parser.add_argument('--lr-scheduler', type=str, default='cosine',
                    help='type of lr scheduler (cosine or multistep)')
parser.add_argument('--lr-decay-milestones', nargs='+', type=int, default=[150000,275000,375000],
                    help='decay learning rate at these milestone epochs (default: [150000,300000])')
parser.add_argument('--optim', default='momentum',
                    help='optimizer, adam, momentum or sgd (default: momentum)')
parser.add_argument('--momentum', type=float, default=0.9,
                    help='SGD momentum (default: 0.9)')
parser.add_argument('--weight-decay', type=float, default=1e-4,
                    help='weight decay (default: 0.0)')
parser.add_argument('--no-similarity-std', action='store_true', default=False,
                    help='disable use of standard deviation in similarity matrix for feature maps')
parser.add_argument('--smoothing', type=float, default=0.1,
                    help='eps for label smoothing')
parser.add_argument('--queue-size', type=int, default=50,
                    help='size of queue for training multiple blocks parallelly')

# model arguments
parser.add_argument('--feat-mult', type=int, default=1,
                    help='multiply number of CNN features with this number (default: 1)')
parser.add_argument('--dim-in-decoder', type=int, default=512,
                    help='input dimension of decoder_y used in predsim loss (default: 512)')
parser.add_argument('--planes', type=int, default=16,
                    help='number of channels in the first conv layer')
parser.add_argument('--nonlin', default='relu',
                    help='nonlinearity, relu or leakyrelu (default: relu)')

# evaluation configurations (for evaluate.py)
parser.add_argument('--fp16', action='store_true', default=False,
                    help='training with fp16')
parser.add_argument('--test', action='store_true', default=False,
                    help='do not train from scratch for evaluation')
parser.add_argument('--valid-freq', type=int, default=100,
                    help='frequency of evaluation on validation split')
parser.add_argument('--eval-base-dir', type=str, default='',
                    help='path to directory where alpha.pt and beta.pt exsit')
parser.add_argument('--eval-continuous', action='store_true',
                    help='train blocks with continuous backward pass')
parser.add_argument('--result-dir', type=str, default='results',
                    help='directory where outputs of evaluation will be saved')
parser.add_argument('--baseline-dec-type', type=str, default='mlp_sr',
                    help='type of baseline (mlp_sr, predsim)')
parser.add_argument('--target-num-blocks', type=int, default=1,
                    help='number of blocks in model')
parser.add_argument('--num-ensemble', default=1, type=int,
                    help='get output from last num-ensemble blocks and do ensemble')


config = parser.parse_args()
config.cuda = not config.no_cuda and torch.cuda.is_available()


