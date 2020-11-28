from __future__ import print_function
import os
import csv
import skimage.io
import argparse
import numpy
import time
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils import data
from torchvision import datasets, models, transforms
import librosa
import soundfile as sf
from scipy import signal
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader
from simple_gcsa.Models import Transformer
from torch.optim.lr_scheduler import ExponentialLR
from collections import OrderedDict
from pypesq import pesq

from tqdm import tqdm

# Reference
# DATA LOADING - LOAD FILE LISTS


def load_data_list(clean_path, noisy_path):

    dataset = {}

    print("Loading files...")
    dataset['innames'] = []
    dataset['outnames'] = []
    dataset['shortnames'] = []

    filelist = os.listdir(noisy_path)
    filelist = [f for f in filelist if f.endswith(".wav")]
    for i in tqdm(filelist):
        dataset['innames'].append("%s/%s" % (noisy_path, i))
        dataset['outnames'].append("%s/%s" % (clean_path, i))
        dataset['shortnames'].append("%s" % (i))

    return dataset

# DATA LOADING - LOAD FILE DATA


def load_data(dataset, max_length):

    dataset['inaudio'] = [None]*len(dataset['innames'])
    dataset['outaudio'] = [None]*len(dataset['outnames'])

    for id in tqdm(range(len(dataset['innames']))):

        if dataset['inaudio'][id] is None:
            inputData, _ = librosa.load(dataset['innames'][id], sr=48000)
            outputData, _ = librosa.load(dataset['outnames'][id], sr=48000)

            in_shape = numpy.shape(inputData)
            if max_length is not None:
                if (in_shape[0] > max_length):
                    inputData = inputData[0:max_length]
                    outputData = outputData[0:max_length]

            dataset['inaudio'][id] = numpy.float32(inputData)
            dataset['outaudio'][id] = numpy.float32(outputData)

    return dataset


class AudioDataset(data.Dataset):
    """
    Audio sample reader.
    """

    def __init__(self, clean_path, noisy_path, max_length=None):
        dataset = load_data_list(clean_path, noisy_path)
        self.dataset = load_data(dataset, max_length)

        self.file_names = dataset['innames']

    def __getitem__(self, idx):
        mixed = torch.from_numpy(
            self.dataset['inaudio'][idx]).type(torch.FloatTensor)
        clean = torch.from_numpy(
            self.dataset['outaudio'][idx]).type(torch.FloatTensor)

        return mixed, clean

    def __len__(self):
        return len(self.file_names)

    def zero_pad_concat(self, inputs):
        max_t = max(inp.shape[0] for inp in inputs)
        shape = (len(inputs), max_t)
        input_mat = numpy.zeros(shape, dtype=numpy.float32)
        for e, inp in enumerate(inputs):
            input_mat[e, :inp.shape[0]] = inp
        return input_mat

    def collate(self, inputs):
        mixeds, cleans = zip(*inputs)
        seq_lens = torch.IntTensor([i.shape[0] for i in mixeds])

        x = torch.FloatTensor(self.zero_pad_concat(mixeds))
        y = torch.FloatTensor(self.zero_pad_concat(cleans))

        batch = [x, y, seq_lens]
        return batch

def SNRCore(clean, est, eps=2e-7):
    def bsum(x): return torch.sum(x, dim=1)
    a = bsum(clean**2)
    b = bsum((clean - est)**2) + eps
    return 10*torch.log10(a/b)


def SNR(clean, est, eps=2e-7):
    return torch.mean(SNRCore(clean, est))


def SegSNR(clean, est, eps=2e-7):
    ssnr = SNRCore(clean, est)
    ssnr = ssnr.clamp(min=-10, max=35)

    return torch.mean(ssnr)


def SDRLoss(clean, est, eps=2e-7):
    def bsum(x): return torch.sum(x, dim=1)
    alpha = bsum(clean*est) / bsum(clean**2)
    alpha = alpha.unsqueeze(1)
    a = bsum((alpha*clean)**2)
    b = bsum((alpha*clean - est)**2)

    return torch.mean(10*torch.log10(a/b))


def wSDRLoss(mixed, clean, clean_est, eps=2e-7):
    # Used on signal level(time-domain). Backprop-able istft should be used.
    # Batched audio inputs shape (N x T) required.
    # Batch preserving sum for convenience.
    def bsum(x): return torch.sum(x, dim=1)

    def mSDRLoss(orig, est):
        # Modified SDR loss, <x, x`> / (||x|| * ||x`||) : L2 Norm.
        # Original SDR Loss: <x, x`>**2 / <x`, x`> (== ||x`||**2)
        #  > Maximize Correlation while producing minimum energy output.
        correlation = bsum(orig * est)
        energies = torch.norm(orig, p=2, dim=1) * torch.norm(est, p=2, dim=1)
        return -(correlation / (energies + eps))

    noise = mixed - clean
    noise_est = mixed - clean_est

    a = bsum(clean**2) / (bsum(clean**2) + bsum(noise**2) + eps)
    wSDR = a * mSDRLoss(clean, clean_est) + (1 - a) * \
        mSDRLoss(noise, noise_est)
    return torch.mean(wSDR)


def calc_dwm(dim):
    mat = numpy.zeros([dim, dim])
    for i in range(dim):
        for j in range(dim):
            mat[i, j] = - (i - j) ** 2
    return torch.Tensor(mat)

def out_result(model, stft, istft, validation_data, device, opt):
    ''' Epoch operation in evaluation phase '''

    model.eval()

    count = 0
    with torch.no_grad():
        for batch in tqdm(validation_data):
            # prepare data
            mixed, clean, seq_len = map(lambda x: x.to(device), batch)

            mixed_stft = stft(mixed)
            mixed_r = mixed_stft[..., 0]
            mixed_i = mixed_stft[..., 1]

            # forward
            mask_r, mask_i = model(
                mixed_r, mixed_i, calc_dwm(mixed_r.shape[2]).to(device))

            output_r = mixed_r*mask_r - mixed_i*mask_i
            output_i = mixed_r*mask_i + mixed_i*mask_r

            output_r = output_r.unsqueeze(-1)
            output_i = output_i.unsqueeze(-1)

            recombined = torch.cat([output_r, output_i], dim=-1)
            output = torch.squeeze(istft(recombined, mixed.shape[1]), dim=1)

            ssnr = SNRCore(clean, output)

            bs = mixed.shape[0]

            for i in range(bs):
                sf.write(
                    'result/{count}_clean.wav'.format(count=count), clean[i].cpu()[0:seq_len[i]], 48000)
                sf.write(
                    'result/{count}_noisy.wav'.format(count=count), mixed[i].cpu()[0:seq_len[i]], 48000)
                sf.write(
                    'result/{count}_output_{ssnr}.wav'.format(count=count, ssnr=ssnr[i]), output[i].cpu()[0:seq_len[i]], 48000)
                count += 1


def prediction(model, stft, istft, dataset, device, opt):
    model.eval()
    total_loss = 0
    total_pesq = 0
    total_ssnr = 0
    total_sdr = 0

    with torch.no_grad():
        for batch in tqdm(dataset):
            # prepare data
            mixed, clean, seq_len = map(lambda x: x.to(device), batch)

            mixed_stft = stft(mixed)
            mixed_r = mixed_stft[..., 0]
            mixed_i = mixed_stft[..., 1]

            # forward
            mask_r, mask_i = model(
                mixed_r, mixed_i, calc_dwm(mixed_r.shape[2]).to(device))

            output_r = mixed_r*mask_r - mixed_i*mask_i
            output_i = mixed_r*mask_i + mixed_i*mask_r

            output_r = output_r.unsqueeze(-1)
            output_i = output_i.unsqueeze(-1)

            recombined = torch.cat([output_r, output_i], dim=-1)
            output = torch.squeeze(istft(recombined, mixed.shape[1]), dim=1)

            # backward and update parameters
            loss = wSDRLoss(mixed, clean, output)
            sdr = SDRLoss(clean, output)
            ssnr = SegSNR(clean, output)

            bs = mixed.shape[0]

            for i in range(bs):
                clean16 = librosa.resample(
                    clean[i].cpu().detach().numpy().copy()[0:seq_len[i]], 48000, 16000)
                output16 = librosa.resample(
                    output[i].cpu().detach().numpy().copy()[0:seq_len[i]], 48000, 16000)

                total_pesq += pesq(clean16, output16, 16000)

            # note keeping
            total_loss += loss.item()
            total_ssnr += ssnr.item()
            total_sdr += sdr.item()

    return total_loss, total_pesq, total_ssnr, total_sdr


def main():
    parser = argparse.ArgumentParser()

    # all-in-1 data pickle or bpe field
    
    parser.add_argument('-model_path', type=str, required=True)
    parser.add_argument('-clean_path', type=str, required=True)
    parser.add_argument('-noisy_path', type=str, required=True)
    parser.add_argument('-no_cuda', action='store_true')

    opt = parser.parse_args()
    opt.cuda = not opt.no_cuda

    device = torch.device('cuda' if opt.cuda else 'cpu')

    #========= Loading Dataset =========#

    checkpoint = torch.load(opt.model_path)

    dataset = AudioDataset(opt.clean_path, opt.noisy_path, max_length=400000)
    data_loader = DataLoader(dataset=dataset, batch_size=checkpoint['settings'].batch_size,
                                  collate_fn=dataset.collate, shuffle=False, num_workers=0)

    opt.d_word_vec = checkpoint['settings'].d_model

    model = Transformer(
        emb_src_trg_weight_sharing=checkpoint['settings'].embs_share_weight,
        d_k=int(checkpoint['settings'].n_fft / 2 / checkpoint['settings'].n_head),
        d_v=int(checkpoint['settings'].n_fft / 2 / checkpoint['settings'].n_head),
        d_model=int(checkpoint['settings'].n_fft / 2 + 1),
        d_inner=checkpoint['settings'].d_inner_hid,
        n_layers=checkpoint['settings'].n_layers,
        n_head=checkpoint['settings'].n_head,
        dropout=checkpoint['settings'].dropout
    ).to(device)

    model.load_state_dict(checkpoint['model'])

    window = torch.hann_window(checkpoint['settings'].n_fft).to(device)
    def stft(x): return torch.stft(x, checkpoint['settings'].n_fft, checkpoint['settings'].hop_length, window=window)

    def istft(x, length): return torch.istft(x,
                                             checkpoint['settings'].n_fft,
                                             checkpoint['settings'].hop_length,
                                             length=length,
                                             window=window)

    def print_performances(header, loss, ssnr):
        print('  - {header:12} loss: {loss: 8.5f},'
              'ssnr: {ssnr}'.format(
                  header=f"({header})", loss=loss, ssnr=ssnr))

    valid_loss, total_pesq, valid_ssnr, valid_sdr = prediction(model, stft, istft, data_loader, device, checkpoint['settings'])
    print_performances('Validation',
                           valid_loss / data_loader.__len__(),
                           valid_ssnr / data_loader.__len__(),
                           )

    print('pesq: {pesq}, sdr: {sdr}'.format(
        pesq=total_pesq,
        sdr=valid_sdr / data_loader.__len__()))

if __name__ == '__main__':
    main()