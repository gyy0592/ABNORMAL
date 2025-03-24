

from __future__ import print_function
import torch
from multiprocessing import Pool, set_start_method
from tqdm import tqdm
import numpy as np
from torch.utils.data import Dataset
import os 

def getDataSet_parallel(extract_func, data_list, num_processes, stats=None, data_type='file'):
    try:
        set_start_method('spawn', force=True)
    except RuntimeError:
        pass  # Ignore the error if the start method has already been set.
    
    if stats is not None:
        means = stats[:, 0]
        std_devs = stats[:, 1]
        args_list = [(file, extract_func, data_type, means, std_devs) for file in data_list]
    else:
        args_list = [(file, extract_func, data_type) for file in data_list]
    
    with Pool(processes=num_processes) as pool:
        # Use tqdm.imap to maintain order and show progress
        processed_data = list(tqdm(pool.imap(preprocess_func_wrapper, args_list), 
                                   total=len(args_list), 
                                   desc="Processing data"))
    
    simpleList = []
    for sequence in processed_data:
        sequence_tensor = torch.tensor(sequence, dtype=torch.float)
        seq_len = sequence_tensor.shape[0]
        x_mark_enc = torch.arange(seq_len).unsqueeze(-1)
        simpleList.append((sequence_tensor, x_mark_enc))
    
    return simple_dataset(simpleList)



def preprocess_func_wrapper(args):
    # This wrapper is needed because pool.imap only takes one argument
    return preprocess_func(*args)

class simple_dataset(Dataset):
    def __init__(self, simpleList):
        self.simpleList = simpleList

    def __len__(self):
        return len(self.simpleList)
    # get item trying to get different frequencies as channels
    def __getitem__(self, idx):
        sequence, x_mark_enc = self.simpleList[idx]
        return sequence, x_mark_enc

    def reshape_and_regenerate(self, new_shape):
        reshaped_list = []
        for sequence, _ in self.simpleList:      
            if new_shape is not None:
                reshaped_sequence = sequence.reshape(new_shape)
            else:
                reshaped_sequence = sequence
            seq_len = reshaped_sequence.shape[0]
            x_mark_enc = torch.arange(seq_len).unsqueeze(-1)
            reshaped_list.append((reshaped_sequence, x_mark_enc))
        # self.simpleList = reshaped_list
        return simple_dataset(reshaped_list)


def preprocess_func(data, extract_func, data_type='file', means=None, std_devs=None, window = 'hann'):
    if data_type == 'file':
        timeSeries = myloadNPZ(data, 'data')
        # If window is None do nothing, if it's 'hann' apply a Hann window
        if means is not None and std_devs is not None:
            sequence = extract_func(timeSeries, means= means, std_devs= std_devs)
        else:
            sequence = extract_func(timeSeries)
    elif data_type == 'value':

        if means is not None and std_devs is not None:
            sequence = extract_func(data, means= means, std_devs= std_devs)
        else:
            sequence = extract_func(data)
    # print(sequence.T.shape)
    return sequence

def myloadNPZ(path, arg='spec_rgb'):
    # Get the file extension
    ext = os.path.splitext(path)[1]

    # Check if the file is .npy
    if ext == '.npy':
        with open(path, 'rb') as f:
            data = np.load(f, allow_pickle=True)
            return data

    # If not .npy, continue with the original functionality for .npz
    with open(path, 'rb') as f:
        npzFile = np.load(f, allow_pickle=True)
        data = npzFile[arg]
        npzFile.close()  # Explicitly close the npz file
        return data