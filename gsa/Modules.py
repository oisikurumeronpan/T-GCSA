import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

__author__ = "@oisikurumeronpan"


class ScaledDotProductAttention(nn.Module):
    ''' Scaled Dot-Product Attention '''

    def __init__(self, temperature, attn_dropout=0.1):
        super().__init__()

        # TODO: add gaussian weight
        self.sigma = nn.Parameter(torch.ones(1))
        self.temperature = temperature
        self.dropout = nn.Dropout(attn_dropout)

    def forward(self, q, k, v, dwm):

        attn = torch.matmul(q / self.temperature, k.transpose(2, 3))

        g = torch.exp(dwm / (self.sigma ** 2))

        attn = torch.abs(attn * g)

        attn = self.dropout(F.softmax(attn, dim=-1))
        output = torch.matmul(attn, v)

        return output, attn
