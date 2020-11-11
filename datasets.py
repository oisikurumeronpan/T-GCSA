import numpy as np
from tqdm import tqdm
import librosa
import os
import csv
import torch
from torch.utils import data
import random

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
            inputData, sr = librosa.load(dataset['innames'][id], sr=48000)
            outputData, sr = librosa.load(dataset['outnames'][id], sr=48000)

            in_shape = np.shape(inputData)
            if max_length is not None:
                if (in_shape[0] > max_length):
                    start = random.randint(0, in_shape[0]-max_length)
                    inputData = inputData[start: start + max_length]
                    outputData = outputData[start: start + max_length]

            dataset['inaudio'][id] = np.float32(inputData)
            dataset['outaudio'][id] = np.float32(outputData)

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
        input_mat = np.zeros(shape, dtype=np.float32)
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
