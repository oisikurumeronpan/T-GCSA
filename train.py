from __future__ import print_function
import os
import skimage.io
import argparse
import numpy
import time
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, models, transforms
import soundfile as sf
from scipy import signal
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader
from datasets import AudioDataset
from gcsa.Models import Transformer, ISTFT
# from gcsa.Optim import ScheduledOptim
from torch.optim.lr_scheduler import ExponentialLR
from collections import OrderedDict
from pypesq import pesq

from tqdm import tqdm


def SDRLoss(clean, est, eps=2e-7):
    def bsum(x): return torch.sum(x, dim=1)
    alpha = bsum(clean*est) / bsum(clean*clean)

    return 10*torch.log10(bsum((alpha*clean)**2)/bsum(alpha*clean - est)**2)


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


def train_epoch(model, stft, istft, training_data, optimizer, opt, device, smoothing):
    ''' Epoch operation in training phase'''

    model.train()
    total_loss = 0

    with tqdm(training_data) as pbar:
        for _, batch in enumerate(pbar):
            # prepare data
            mixed, clean, _ = map(lambda x: x.to(device), batch)

            mixed_stft = stft(mixed)
            mixed_r, mixed_i = mixed_stft[..., 0], mixed_stft[..., 1]

            # forward
            optimizer.zero_grad()
            mask_r, mask_i = model(
                mixed_r, mixed_i, calc_dwm(mixed_r.shape[2]).to(device))

            output_r, output_i = mixed_r*mask_r - mixed_i * \
                mask_i, mixed_r*mask_i + mixed_i*mask_r

            output_r = output_r.unsqueeze(-1)
            output_i = output_i.unsqueeze(-1)

            recombined = torch.cat([output_r, output_i], dim=-1)

            output = torch.squeeze(istft(recombined, mixed.shape[1]), dim=1)

            # backward and update parameters
            # loss = wSDRLoss(mixed, clean, output)
            loss = SDRLoss(clean, output)
            loss.backward()
            optimizer.step()

            # bs = mixed.shape[0]
            # avg_pesq = 0

            # for i in range(bs):
            #     avg_pesq += pesq(clean[i].cpu().detach().numpy(),
            #                      output[i].cpu().detach().numpy(), 16000)

            # avg_pesq = avg_pesq / bs

            # note keeping
            total_loss += loss.item()
            pbar.set_postfix(OrderedDict(loss=math.exp(loss.item())))

    return total_loss


def eval_epoch(model, stft, istft, validation_data, device, opt):
    ''' Epoch operation in evaluation phase '''

    model.eval()
    total_loss = 0
    total_pesq = 0

    with torch.no_grad():
        for batch in tqdm(validation_data):
            # prepare data
            mixed, clean, _ = map(lambda x: x.to(device), batch)

            mixed_stft = stft(mixed)
            mixed_r, mixed_i = mixed_stft[..., 0], mixed_stft[..., 1]

            # forward
            mask_r, mask_i = model(
                mixed_r, mixed_i, calc_dwm(mixed_r.shape[2]).to(device))

            output_r, output_i = mixed_r*mask_r - mixed_i * \
                mask_i, mixed_r*mask_i + mixed_i*mask_r

            output_r = output_r.unsqueeze(-1)
            output_i = output_i.unsqueeze(-1)

            recombined = torch.cat([output_r, output_i], dim=-1)
            output = torch.squeeze(istft(recombined, mixed.shape[1]), dim=1)

            # backward and update parameters
            loss = wSDRLoss(mixed, clean, output)

            bs = mixed.shape[0]

            for i in range(bs):
                total_pesq += pesq(clean[i].cpu(), output[i].cpu(), 16000)

            # note keeping
            total_loss += loss.item()

    return total_loss, total_pesq


def train(model, stft, istft, training_data, validation_data, optimizer, scheduler, device, opt):
    ''' Start training '''

    log_train_file, log_valid_file = None, None

    if opt.log:
        log_train_file = opt.log + '.train.log'
        log_valid_file = opt.log + '.valid.log'

        print('[Info] Training performance will be written to file: {} and {}'.format(
            log_train_file, log_valid_file))

        with open(log_train_file, 'w') as log_tf, open(log_valid_file, 'w') as log_vf:
            log_tf.write('epoch,loss,ppl,accuracy\n')
            log_vf.write('epoch,loss,ppl,accuracy\n')

    def print_performances(header, loss, start_time):
        print('  - {header:12} loss: {loss: 8.5f},'
              'elapse: {elapse:3.3f} min'.format(
                  header=f"({header})", loss=math.exp(min(loss, 100)),
                  elapse=(time.time()-start_time)/60))

    #valid_accus = []
    valid_losses = []
    for epoch_i in range(opt.epoch):
        print('[ Epoch', epoch_i, ']')

        start = time.time()
        train_loss = train_epoch(
            model, stft, istft, training_data, optimizer, opt, device, smoothing=opt.label_smoothing)
        print_performances('Training', train_loss /
                           training_data.__len__(), start)

        start = time.time()
        valid_loss, total_pesq = eval_epoch(
            model, stft, istft, validation_data, device, opt)
        print_performances('Validation', valid_loss /
                           validation_data.__len__(), start,)
        print('pesq: ', total_pesq)

        valid_losses += [valid_loss]

        checkpoint = {'epoch': epoch_i, 'settings': opt,
                      'model': model.state_dict()}

        scheduler.step()

        if opt.save_model:
            if opt.save_mode == 'all':
                model_name = opt.save_model + \
                    '_loss_{accu:3.3f}.chkpt'.format(accu=100*valid_loss)
                torch.save(checkpoint, model_name)
            elif opt.save_mode == 'best':
                model_name = opt.save_model + '.chkpt'
                if valid_loss <= min(valid_losses):
                    torch.save(checkpoint, model_name)
                    print('    - [Info] The checkpoint file has been updated.')

        if log_train_file and log_valid_file:
            with open(log_train_file, 'a') as log_tf, open(log_valid_file, 'a') as log_vf:
                log_tf.write('{epoch},{loss: 8.5f},{loss: 8.5f},\n'.format(
                    epoch=epoch_i, loss=train_loss,
                    ppl=math.exp(min(train_loss, 100))))
                log_vf.write('{epoch},{loss: 8.5f},{ppl: 8.5f},\n'.format(
                    epoch=epoch_i, loss=valid_loss,
                    ppl=math.exp(min(valid_loss, 100))))


def outputWavDatas(args, model, device, loader, sl, target_):
    target_Zxx = signal.stft(target_, fs=sl)[2]
    model.eval()
    with torch.no_grad():
        samples = next(loader.__iter__())
        input_img, teach_img = samples['input_img'].to(
            device), samples['teach_img'].to(device)

        input_ = model(input_img)
        teach_ = model(teach_img)

        input = input_img.cpu()[0, 0, :, :]
        teach = teach_img.cpu()[0, 0, :, :]
        input_ = input_.cpu()[0, 0, :, :]
        teach_ = teach_.cpu()[0, 0, :, :]

        test = input_.numpy()*target_Zxx[0:128, 0:520]

        print('Start Output')
        _, Input = signal.istft(input, fs=sl)
        _, Teach = signal.istft(teach, fs=sl)
        _, Input_ = signal.istft(input_, fs=sl)
        _, Teach_ = signal.istft(teach_, fs=sl)

        _, Test = signal.istft(test, fs=sl)

        sf.write('input.wav', Input, sl)
        sf.write('teach.wav', Teach, sl)
        sf.write('input_.wav', Input_, sl)
        sf.write('teach_.wav', Teach_, sl)

        sf.write('test.wav', Test, sl)


def main():
    parser = argparse.ArgumentParser()

    # all-in-1 data pickle or bpe field
    parser.add_argument('-data_pkl', default=None)

    parser.add_argument('-train_path', default=None)   # bpe encoded data
    parser.add_argument('-val_path', default=None)     # bpe encoded data

    parser.add_argument('-epoch', type=int, default=10)
    parser.add_argument('-b', '--batch_size', type=int, default=12)

    parser.add_argument('-d_model', type=int, default=512)
    parser.add_argument('-d_inner_hid', type=int, default=2048)
    parser.add_argument('-d_k', type=int, default=64)
    parser.add_argument('-d_v', type=int, default=64)

    parser.add_argument('-n_head', type=int, default=8)
    parser.add_argument('-n_layers', type=int, default=6)
    parser.add_argument('-warmup', '--n_warmup_steps', type=int, default=4000)

    parser.add_argument('-dropout', type=float, default=0.1)
    parser.add_argument('-embs_share_weight', action='store_true')
    parser.add_argument('-proj_share_weight', action='store_true')

    parser.add_argument('-log', default=None)
    parser.add_argument('-save_model', default=None)
    parser.add_argument('-save_mode', type=str,
                        choices=['all', 'best'], default='best')

    parser.add_argument('-no_cuda', action='store_true')
    parser.add_argument('-label_smoothing', action='store_true')

    parser.add_argument('-n_fft', type=int, default=1024)
    parser.add_argument('-hop_length', type=int, default=512)
    parser.add_argument('-max_length', type=int, default=100000)

    opt = parser.parse_args()
    opt.cuda = not opt.no_cuda
    opt.d_word_vec = opt.d_model

    device = torch.device('cuda' if opt.cuda else 'cpu')

    #========= Loading Dataset =========#

    train_dataset = AudioDataset(data_type='train', max_length=opt.max_length)
    test_dataset = AudioDataset(data_type='val')
    train_data_loader = DataLoader(dataset=train_dataset, batch_size=opt.batch_size,
                                   collate_fn=train_dataset.collate, shuffle=True, num_workers=0)
    test_data_loader = DataLoader(dataset=test_dataset, batch_size=opt.batch_size,
                                  collate_fn=test_dataset.collate, shuffle=False, num_workers=0)

    model = Transformer(
        emb_src_trg_weight_sharing=opt.embs_share_weight,
        d_k=int(opt.n_fft / 2 / opt.n_head),
        d_v=int(opt.n_fft / 2 / opt.n_head),
        d_model=int(opt.n_fft / 2 + 1),
        d_inner=opt.d_inner_hid,
        n_layers=opt.n_layers,
        n_head=opt.n_head,
        dropout=opt.dropout
    ).to(device)

    window = torch.hann_window(opt.n_fft).to(device)
    def stft(x): return torch.stft(x, opt.n_fft, opt.hop_length, window=window)
    # istft = ISTFT(opt.n_fft, opt.hop_length, window='hanning').to(device)

    def istft(x, length): return torch.istft(x,
                                             opt.n_fft,
                                             opt.hop_length,
                                             length=length,
                                             window=window)

    # optimizer = ScheduledOptim(
    #     optim.Adam(model.parameters(), betas=(0.9, 0.98), eps=1e-09),
    #     2.0, opt.d_model, opt.n_warmup_steps
    # )

    optimizer = optim.Adam(model.parameters(), lr=1e-3)

    scheduler = ExponentialLR(optimizer, 0.95)

    train(
        model=model,
        stft=stft,
        istft=istft,
        training_data=train_data_loader,
        validation_data=test_data_loader,
        optimizer=optimizer,
        scheduler=scheduler,
        device=device,
        opt=opt,
    )


if __name__ == '__main__':
    main()
