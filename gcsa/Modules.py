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
        self.dropout_r = nn.Dropout(attn_dropout)
        self.dropout_i = nn.Dropout(attn_dropout)

    def forward(self, q_r, q_i, k_r, k_i, v_r, v_i, dwm):

        attn_r = torch.matmul(q_r / self.temperature, k_r.transpose(2, 3)) + \
            torch.matmul(q_i / self.temperature, k_i.transpose(2, 3))
        attn_i = torch.matmul(q_i / self.temperature, k_r.transpose(2, 3)) - \
            torch.matmul(q_r / self.temperature, k_i.transpose(2, 3))

        # attn_r = torch.matmul(q_r / self.temperature, k_r.transpose(2, 3))
        # attn_i = torch.matmul(q_i / self.temperature, k_i.transpose(2, 3))

        g = torch.exp(dwm / (self.sigma ** 2))

        attn_r = torch.abs(attn_r * g)
        attn_i = torch.abs(attn_i * g)

        attn_r = self.dropout_r(F.softmax(attn_r, dim=-1))
        attn_i = self.dropout_i(F.softmax(attn_i, dim=-1))
        output_r, output_i = torch.matmul(
            attn_r, v_r), torch.matmul(attn_i, v_i)

        return output_r, output_i, attn_r, attn_i
