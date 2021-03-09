import torch
import torch.nn as nn
import torch.nn.functional as F
from itertools import chain

from config import config

from module import *
from decoder import *
from loss import *
from dataset import DATASET_CONFIGS
import torch.cuda.amp as amp
from contextlib import ExitStack

class BaseNet(nn.Module):
    def __init__(self):
        super(BaseNet, self).__init__()
        cfg_data = DATASET_CONFIGS[config.dataset]
        self.size, self.channels, self.classes = cfg_data['size'], cfg_data['channels'], cfg_data['classes']
        self.moduleList = nn.ModuleList()

    def forward(self, x, isolate_grad=False, save_act=True, idx_eval_loss=None):
        out = x
        block_idx = 0

        for i, m in enumerate(self.moduleList):
            if isinstance(m, LocalLossBlock):
                out = m(out, save_act=save_act)
                if isolate_grad or (idx_eval_loss is not None and block_idx not in idx_eval_loss):
                    out = out.detach()
                block_idx += 1
            else:
                out = m(out)

        out = m.decoder(out)
        if isinstance(out, tuple):
            out = out[0]

        return out


    def get_loss(self, y, weights=None, categorical=False, idx_eval_loss=None, scale=None):
        if isinstance(weights, torch.Tensor) and weights.size(-1) == 0: weights = None
        if scale is None: scale = [1.]*self.num_blocks
        losses = []
        block_idx = 0
        for m in self.moduleList:
            if isinstance(m, LocalLossBlock):
                if idx_eval_loss is not None and not (block_idx in idx_eval_loss):
                    losses.append(0.)
                elif weights is None or not isinstance(m.decoder, MixedPredConvLightDecoder):
                    losses.append(m.get_loss(y, scale=scale[block_idx]))
                else:
                    losses.append(m.get_loss(y, weights[block_idx], categorical=categorical, scale=scale[block_idx]))

                block_idx += 1

        return losses


    @property
    def num_blocks(self):
        count = 0
        for m in self.moduleList:
            if isinstance(m, LocalLossBlock):
                count += 1
        return count

    @property
    def num_decoders(self):
        nd = 0
        for m in self.moduleList:
            if isinstance(m, LocalLossBlock):
                nd = m.decoder.num_decoders
                break

        return nd

    @property
    def mapping_from_header_index(self):
        ''' Mapping from header index to module indices'''
        index_table = [[] for _ in range(self.num_blocks)]
        curr_block_idx = 0
        for i, m in enumerate(self.moduleList):
            index_table[curr_block_idx].append(i)
            if isinstance(m, LocalLossBlock):
                curr_block_idx += 1

        return index_table

    def get_blocks(self):
        blocks = []
        mapping = self.mapping_from_header_index
        for i in range(self.num_blocks):
            block = BaseNet()
            block.moduleList.extend([self.moduleList[j] for j in mapping[i]])
            blocks.append(block)

        assert id(blocks[0].moduleList[0]) == id(self.moduleList[0])
        return blocks

    def get_decoder(self, block_idx):
        i = 0
        for m in self.moduleList:
            if isinstance(m, LocalLossBlock):
                if i == block_idx:
                    return m.decoder
                else:
                    i += 1
        raise IndexError

    def block_parameters(self, block_idx):
        module_indices = self.mapping_from_header_index[block_idx]
        params = []
        for i in module_indices:
            params += [self.moduleList[i].parameters()]

        return chain(*params)

    def block_state_dict(self, block_idx):
        ''' Get state_dict of modules belonging to block_idx-th block'''
        if block_idx < self.num_blocks:
            module_indices = self.mapping_from_header_index[block_idx]
            state_dicts = [self.moduleList[i].state_dict() for i in module_indices]
        else:
            state_dicts = self.classifier.state_dict()

        return state_dicts

    def load_block_state_dict(self, state_dicts, block_idx):
        ''' Load state_dict of modules belonging to block_idx-th block'''
        if block_idx < self.num_blocks:
            module_indices = self.mapping_from_header_index[block_idx]
            for i, module_index in enumerate(module_indices):
                self.moduleList[module_index].load_state_dict(state_dicts[i])
        else:
            self.classifier.load_state_dict(state_dicts)

        return None

    def update_model(self, alpha, beta=None, num_blocks=0):
        if isinstance(beta, torch.Tensor) and beta.size(-1) == 0: beta = None
        if num_blocks == 0:
            is_allow_upper_grad = alpha.data.max(1)[1].bool().tolist()
        else:
            alpha_softmax = F.softmax(alpha, dim=1)
            block_indices = alpha_softmax[:, 0].topk(num_blocks-1).indices.tolist()
            is_allow_upper_grad = [i not in block_indices for i in range(self.num_blocks-1)]

        block_config = [str(i) for i in range(self.num_blocks-1) if not is_allow_upper_grad[i]]
        block_config.append(str(self.num_blocks-1))

        if beta is not None:
            decoder_configs = []

        print("Update model with block idx {}".format(', '.join(block_config)))
        curr_block_idx = 0
        num_blocks = self.num_blocks
        for i, m in enumerate(self.moduleList):
            if isinstance(m, LocalLossBlock):
                if curr_block_idx < num_blocks - 1 and is_allow_upper_grad[curr_block_idx]:
                        self.moduleList[i] = m.encoder
                elif curr_block_idx < num_blocks - 1 and beta is not None:
                    best_decoder_idx = beta[curr_block_idx, :].max(0)[1].item()
                    decoder_configs.append(self.moduleList[i].decoder.decoder_configs[best_decoder_idx])
                    self.moduleList[i].decoder = self.moduleList[i].decoder.decoders[best_decoder_idx]

                curr_block_idx += 1

        if beta is not None:
            return block_config, decoder_configs
        else:
            return block_config, None


