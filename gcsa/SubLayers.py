import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from gcsa.Modules import ScaledDotProductAttention


__author__ = "@oisikurumeronpan"


class MultiHeadAttention(nn.Module):
    ''' Multi-Head Attention module '''

    def __init__(self, n_head, d_model, d_k, d_v, dropout=0.1):
        super().__init__()

        self.n_head = n_head
        self.d_k = d_k
        self.d_v = d_v

        self.w_qrs = nn.Linear(d_model, n_head * d_k, bias=False)
        self.w_qis = nn.Linear(d_model, n_head * d_k, bias=False)
        self.w_krs = nn.Linear(d_model, n_head * d_k, bias=False)
        self.w_kis = nn.Linear(d_model, n_head * d_k, bias=False)
        self.w_vrs = nn.Linear(d_model, n_head * d_v, bias=False)
        self.w_vis = nn.Linear(d_model, n_head * d_v, bias=False)
        self.frc = nn.Linear(n_head * d_v, d_model, bias=False)
        self.fic = nn.Linear(n_head * d_v, d_model, bias=False)

        self.attention = ScaledDotProductAttention(temperature=d_k ** 0.5)

        self.dropout_r = nn.Dropout(dropout)
        self.dropout_i = nn.Dropout(dropout)
        self.layer_norm_r = nn.LayerNorm(d_model, eps=1e-6)
        self.layer_norm_i = nn.LayerNorm(d_model, eps=1e-6)

    def forward(self, q_r, q_i, k_r, k_i, v_r, v_i, dwm):

        d_k, d_v, n_head = self.d_k, self.d_v, self.n_head
        sz_b, len_q, len_k, len_v = q_r.size(
            0), q_r.size(1), k_r.size(1), v_r.size(1)

        residual_r = q_r
        residual_i = q_i

        # Pass through the pre-attention projection: b x lq x (n*dv)
        # Separate different heads: b x lq x n x dv
        q_r = self.w_qrs(q_r).view(sz_b, len_q, n_head, d_k)
        q_i = self.w_qis(q_i).view(sz_b, len_q, n_head, d_k)
        k_r = self.w_krs(k_r).view(sz_b, len_k, n_head, d_k)
        k_i = self.w_kis(k_i).view(sz_b, len_k, n_head, d_k)
        v_r = self.w_vrs(v_r).view(sz_b, len_v, n_head, d_v)
        v_i = self.w_vis(v_i).view(sz_b, len_v, n_head, d_v)

        # Transpose for attention dot product: b x n x lq x dv
        q_r, k_r, v_r = q_r.transpose(
            1, 2), k_r.transpose(1, 2), v_r.transpose(1, 2)
        q_i, k_i, v_i = q_i.transpose(
            1, 2), k_i.transpose(1, 2), v_i.transpose(1, 2)

        q_r, q_i, attn_r, attn_i = self.attention(
            q_r, q_i, k_r, k_i, v_r, v_i, dwm)

        # Transpose to move the head dimension back: b x lq x n x dv
        # Combine the last two dimensions to concatenate all the heads together: b x lq x (n*dv)
        q_r = q_r.transpose(1, 2).contiguous().view(sz_b, len_q, -1)
        q_i = q_i.transpose(1, 2).contiguous().view(sz_b, len_q, -1)
        q_r = self.dropout_r(self.frc(q_r))
        q_i = self.dropout_i(self.fic(q_i))
        q_r += residual_r
        q_i += residual_i

        q_r = self.layer_norm_r(q_r)
        q_i = self.layer_norm_i(q_i)

        return q_r, q_i, attn_r, attn_i


class PositionwiseFeedForward(nn.Module):
    ''' A two-feed-forward-layer module '''

    def __init__(self, d_in, d_hid, dropout=0.1):
        super().__init__()
        self.w_1r = nn.Linear(d_in, d_hid)  # position-wise
        self.w_1i = nn.Linear(d_in, d_hid)  # position-wise
        self.w_2r = nn.Linear(d_hid, d_in)  # position-wise
        self.w_2i = nn.Linear(d_hid, d_in)  # position-wise
        self.layer_norm_r = nn.LayerNorm(d_in, eps=1e-6)
        self.layer_norm_i = nn.LayerNorm(d_in, eps=1e-6)
        self.dropout_r = nn.Dropout(0.5)
        self.dropout_i = nn.Dropout(0.5)

    def forward(self, x_r, x_i):

        x_r1 = F.relu(self.w_1r(x_r) - self.w_1i(x_i))
        x_i1 = F.relu(self.w_1i(x_r) + self.w_1r(x_i))
        x_r2 = self.dropout_r(self.w_2r(x_r1) - self.w_2i(x_i1))
        x_i2 = self.dropout_i(self.w_2i(x_r1) + self.w_2r(x_i1))
        x_r2 += x_r
        x_i2 += x_i

        x_r2 = self.layer_norm_r(x_r2)
        x_i2 = self.layer_norm_i(x_i2)

        return x_r2, x_i2
