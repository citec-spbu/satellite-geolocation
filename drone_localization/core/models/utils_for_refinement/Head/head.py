import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from .ChannelEmbedding import ChannelEmbedding



def make_head(opt):
    head_model = Head(opt)
    return head_model


class Head(nn.Module):
    def __init__(self, opt):
        super().__init__()
        self.opt = opt
        self.head = self.init_head(opt.model["head"])

    def init_head(self, head_opt):
        head = head_opt.pop("type")
        if head == "ChannelEmbedding":
            head_model = ChannelEmbedding(**head_opt)
        else:
            raise NameError("{} not in the head list!!!".format(head))
        return head_model

    def forward(self, z, x):
        return self.head(z, x)
