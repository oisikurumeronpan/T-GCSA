import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import scipy.signal
import librosa
from gcsa.Layers import EncoderLayer


__author__ = "@oisikurumeronpan"


class Encoder(nn.Module):
    ''' A encoder model with self attention mechanism. '''

    def __init__(
            self, n_layers, n_head, d_k, d_v,
            d_model, d_inner, dropout=0.1):

        super().__init__()

        self.layer_stack = nn.ModuleList([
            EncoderLayer(d_model, d_inner, n_head, d_k, d_v, dropout=dropout)
            for _ in range(n_layers)])

    def forward(self, src_seq_r, src_seq_i, dwm, return_attns=False):

        enc_slf_attn_list_r = []
        enc_slf_attn_list_i = []

        # -- Forward

        enc_output_r, enc_output_i = src_seq_r, src_seq_i

        for enc_layer in self.layer_stack:
            enc_output_r, enc_output_i, enc_slf_attn_r, enc_slf_attn_i = enc_layer(
                enc_output_r, enc_output_i, dwm)
            enc_slf_attn_list_r += [enc_slf_attn_r] if return_attns else []
            enc_slf_attn_list_i += [enc_slf_attn_i] if return_attns else []

        if return_attns:
            return enc_output_r, enc_output_i, enc_slf_attn_list_r, enc_slf_attn_list_i
        return enc_output_r, enc_output_i


class Transformer(nn.Module):
    ''' A sequence to sequence model with attention mechanism. '''

    def __init__(
            self, d_model=512, d_inner=2048,
            n_layers=6, n_head=8, d_k=64, d_v=64, dropout=0.1,
            emb_src_trg_weight_sharing=True):

        super().__init__()

        self.encoder = Encoder(
            d_model=d_model,
            d_inner=d_inner,
            n_layers=n_layers,
            n_head=n_head,
            d_k=d_k,
            d_v=d_v,
            dropout=dropout,
        )

        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

        if emb_src_trg_weight_sharing:
            self.encoder.src_word_emb.weight = self.decoder.trg_word_emb.weight

    def forward(self, src_seq_r, src_seq_i, dwm):
        src_seq_r, src_seq_i = src_seq_r.transpose(
            1, 2), src_seq_i.transpose(1, 2)  # B*D*T -> B*T*D
        enc_output_r, enc_output_i, * \
            _ = self.encoder(src_seq_r, src_seq_i, dwm)
        enc_output_r, enc_output_i = enc_output_r.transpose(
            1, 2), enc_output_i.transpose(1, 2)  # B*T*D -> B*D*T

        return enc_output_r, enc_output_i


class ISTFT(torch.nn.Module):
    def __init__(self, filter_length=1024, hop_length=512, window='hanning', center=True):
        super(ISTFT, self).__init__()

        self.filter_length = filter_length
        self.hop_length = hop_length
        self.center = center

        win_cof = scipy.signal.get_window(window, filter_length)
        self.inv_win = self.inverse_stft_window(win_cof, hop_length)

        fourier_basis = np.fft.fft(np.eye(self.filter_length))
        cutoff = int((self.filter_length / 2 + 1))
        fourier_basis = np.vstack([np.real(fourier_basis[:cutoff, :]),
                                   np.imag(fourier_basis[:cutoff, :])])
        inverse_basis = torch.FloatTensor(self.inv_win *
                                          np.linalg.pinv(fourier_basis).T[:, None, :])

        self.register_buffer('inverse_basis', inverse_basis.float())

    # Use equation 8 from Griffin, Lim.
    # Paper: "Signal Estimation from Modified Short-Time Fourier Transform"
    # Reference implementation: https://github.com/tensorflow/tensorflow/blob/master/tensorflow/python/ops/signal/spectral_ops.py
    # librosa use equation 6 from paper: https://github.com/librosa/librosa/blob/0dcd53f462db124ed3f54edf2334f28738d2ecc6/librosa/core/spectrum.py#L302-L311
    def inverse_stft_window(self, window, hop_length):
        window_length = len(window)
        denom = window ** 2
        overlaps = -(-window_length // hop_length)  # Ceiling division.
        denom = np.pad(denom, (0, overlaps * hop_length -
                               window_length), 'constant')
        denom = np.reshape(denom, (overlaps, hop_length)).sum(0)
        denom = np.tile(denom, (overlaps, 1)).reshape(overlaps * hop_length)
        return window / denom[:window_length]

    def forward(self, real_part, imag_part, length=None):
        if (real_part.dim() == 2):
            real_part = real_part.unsqueeze(0)
            imag_part = imag_part.unsqueeze(0)

        recombined = torch.cat([real_part, imag_part], dim=1)

        inverse_transform = F.conv_transpose1d(recombined,
                                               self.inverse_basis,
                                               stride=self.hop_length,
                                               padding=0)

        padded = int(self.filter_length // 2)
        if length is None:
            if self.center:
                inverse_transform = inverse_transform[:, :, padded:-padded]
        else:
            if self.center:
                inverse_transform = inverse_transform[:, :, padded:]
            inverse_transform = inverse_transform[:, :, :length]

        return inverse_transform
