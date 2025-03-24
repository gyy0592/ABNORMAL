

from __future__ import print_function
import os
import yaml
import torch
from torch.utils.data import DataLoader
from utils.myclass.config_class import FileConfig, Config
from utils.myclass.mymodel import SignalReconstructionModel
from utils.myfuncs.BarryStatDetDataset import getDataSet_parallel
from utils.myfuncs.load_model import load_model
from utils.myfuncs.dataset import load_dataset, generate_dataset
from utils.myfuncs.histSeries import generate_subbandsv2
from utils.myfuncs.postProcess import calculate_loss_and_snr


def obtain_lossDis(data, saved_model_path = None, tempFile_path= None, lines_freq = None, lines_band = None, freq_range = [30, 1700], nFreq_multi = 100):
    config_path = 'config/config.yaml'
    config_data = yaml.safe_load(open(config_path))
    # Create instances of the FileConfig and Config classes
    fileConfig = FileConfig(config_data['fileConfig'])
    configs = Config(config_data['Config'])
    if not os.path.exists(fileConfig.weights_dir):
        os.makedirs(fileConfig.weights_dir)
    # Example usage
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    # test_data1 = getDataSet_parallel(extract_func=configs[0].extract_func, data_list=data, num_processes=10, data_type='value')
    if tempFile_path is None:
        print("no tempFile_path, generating datasets in memory")
        test_data1 = getDataSet_parallel(extract_func=configs[0].extract_func, data_list=data, num_processes=10, data_type='value')
        test_data2 = getDataSet_parallel(extract_func=configs[1].extract_func, data_list=data, num_processes=10, data_type='value')
    else:
        test_data1_file = f'{tempFile_path}1.path'
        test_data2_file = f'{tempFile_path}2.path'
        # Generate the datasets
        test_data1_gen = generate_dataset(test_data1_file, fileConfig, 'temp', getDataSet_parallel, 1, extract_func = configs[0].extract_func, data_list = data, num_processes=5, data_type = 'value')
        # test_data1_gen = generate_dataset(test_data1_file, fileConfig, 'train', getDataSet, configs[0].extract_func, test_list)
        test_data2_gen  = generate_dataset(test_data2_file,fileConfig, 'temp', getDataSet_parallel, 2, extract_func = configs[1].extract_func, data_list = data, num_processes=10, data_type= 'value')

        # Load or generate datasets
        test_data1 = load_dataset(test_data1_file, fileConfig,'temp')
        test_data2 = load_dataset(test_data2_file,fileConfig,'temp')
    # Create DataLoaders
    a,b = test_data1[0]
    c,d = test_data2[0]
    configs[0].seq_len = len(a)
    configs[1].seq_len = len(c)
    configs[0].pred_len = len(a)
    configs[1].pred_len = len(c)
    batch_size = fileConfig.test_batch_size
    test_loader1 = DataLoader(dataset=test_data1, batch_size=batch_size, shuffle=False)
    test_loader2 = DataLoader(dataset=test_data2, batch_size=batch_size, shuffle=False)
    if saved_model_path is None:
        saved_model_path = fileConfig.weights_dir + 'best_weights.pth'  # Update with your model path
    else:
        saved_model_path = saved_model_path
    # Load the model
    model = load_model(configs, SignalReconstructionModel, device, saved_model_path)
    if lines_freq is not None:
        freq_start = freq_range[0]
        freq_end = freq_range[1]
        # Generating subBands array
        subBands, _ = generate_subbandsv2(freq_start, freq_end, nFreq_multi, 0.75, 12)
        subBands2remove = []
        
        for line_freq in lines_freq:
            for idx, subBand in enumerate(subBands):
                # Check if line_freq falls within the subBand range
                # if subBand[0] <= line_freq <= subBand[1]:
                if (subBand[0] <= line_freq - 1 <= subBand[1]) or (subBand[0] <= line_freq + 1 <= subBand[1]):
                    subBands2remove.append(idx)
        # keep the unique ones
        subBands2remove = list(set(subBands2remove))

    if lines_band is not None:
        freq_start = freq_range[0]
        freq_end = freq_range[1]
        # Generating subBands array
        subBands, _ = generate_subbandsv2(freq_start, freq_end, nFreq_multi, 0.75, 12)
        subBands2remove = []
        
        for line_range in lines_band:
            for idx, subBand in enumerate(subBands):
                # Check if any part of the line_range overlaps with the subBand
                if (subBand[0] <= line_range[0] <= subBand[1]) or (subBand[0] <= line_range[1] <= subBand[1]):
                    subBands2remove.append(idx)
        # keep the unique ones
        subBands2remove = list(set(subBands2remove))

    if lines_band is not None and len(subBands2remove) > 0:
        goodLoss, _, _, _, _ = calculate_loss_and_snr(test_loader1, test_loader2, data, model, device, subBands2remove=subBands2remove)
    
    elif lines_freq is not None and len(subBands2remove) > 0:
        goodLoss, _, _, _, _ = calculate_loss_and_snr(test_loader1, test_loader2, data, model, device, subBands2remove = subBands2remove)
    else:
        goodLoss, _, _, _, _ = calculate_loss_and_snr(test_loader1, test_loader2, data, model, device)
    return goodLoss



