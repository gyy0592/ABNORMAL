
from __future__ import print_function
import os
import random
import torch
import sys
import yaml
import numpy as np
import os 
import pickle
from torch.utils.data import DataLoader, Sampler
from utils.myfuncs.BarryStatDetDataset import getDataSet_parallel
from utils.myclass.config_class import FileConfig, Config
from utils.myfuncs.mytrain import mytrain
from utils.myfuncs.dataset import load_dataset, generate_dataset
from utils.myclass.loss_class import trainLoss, myTestLoss



class Logger:
    def __init__(self, log_file):
        self.terminal = sys.stdout
        if not os.path.exists(os.path.dirname(log_file)):
            os.makedirs(os.path.dirname(log_file))
        self.log = open(log_file, 'w')

    def write(self, message):
        self.terminal.write(message)  # Write to terminal
        self.log.write(message)       # Write to log file
        self.log.flush()              # Force flush to log file after each write

    def flush(self):
        pass  # To satisfy the `sys.stdout` and `sys.stderr` API
if __name__ == '__main__':

    config_path = 'config/config.yaml'

    # Assume 'config_path' is defined and points to the YAML configuration file
    with open(config_path, 'r') as file:
        config_data = yaml.safe_load(file)


    # Create instances of the FileConfig and Config classes
    fileConfig = FileConfig(config_data['fileConfig'])
    configs = Config(config_data['Config'])

    # Open a log file in write mode
    # log_file = open(fileConfig.training_log, 'w')
    # Redirect stdout and stderr to log file
    sys.stdout = Logger(fileConfig.training_log)
    sys.stderr = Logger(fileConfig.training_log)


    if not os.path.exists(fileConfig.weights_dir):
        os.makedirs(fileConfig.weights_dir)

    # print(configs.configs[0].d_model)

    # Accessing nested configurations
    print((configs[0].extract_func))
    print(configs[1].d_model)

    for configii in configs :
        print(configii.d_model)

    my_loss_func = trainLoss()

    criterion = trainLoss()


    # ### gpt updated trainning function


    seed = 42
    def seed_everything(seed):
        random.seed(seed)
        os.environ['PYTHONHASHSEED'] = str(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True

    seed_everything(seed)
    # testLossFunc = torch.nn.MSELoss(reduction='none')
    testLossFunc = myTestLoss()

    print(f"Torch: {torch.__version__}")


    # ### Training parameters


    # Training settings
    batch_size = fileConfig.train_batch_size
    epochs = fileConfig.train_number_epochs
    # lr = 3e-5
    # gamma = 0.7
    # gamma = 0



    device = 'cuda'


    # ## Load Data
    # temp = np.load('config/LIGO_stat_paths.npz')['test_list'].tolist()
    train_list = np.load(fileConfig.dataset_pathsFile)['train_list'].tolist()
    test_list = np.load(fileConfig.dataset_pathsFile)['test_list'].tolist()
    # train_list = temp
    # print first 5 elemtns of training set
    print(train_list[:5])

    labels = []

    for path in train_list:
        # Get the filename from the path
        filename = os.path.basename(path)
        
        # Check for the presence of the labels in the filename
        if 'stationaryNoise' in filename:
            label = 'stationaryNoise'
        elif 'nonStationaryNoise' in filename:
            label = 'nonStationaryNoise'
        else:
            label = 'Unknown'
        
        labels.append(label)




    # ### Plot the hist gram of at the certain frequency
    # 
    # 1. convert the data from 3d image to 2d data

    # ## Split


    # train_list, valid_list = train_test_split(train_list, 
    #                                         test_size=0.35,
    #                                         random_state=seed)


    print(f"Train Data: {len(train_list)}")
    # print(f"Validation Data: {len(valid_list)}")
    print(f"Test Data: {len(test_list)}")


    # File names for saved datasets
    test_data1_file = 'test_data1.pth'
    test_data2_file = 'test_data2.pth'
    train_data1_file = 'train_data1.pth'
    train_data2_file = 'train_data2.pth'

    # Generate the datasets
    test_data1_gen = generate_dataset(test_data1_file, fileConfig, 'train', getDataSet_parallel, 1, extract_func = configs[0].extract_func, data_list = test_list, num_processes=5)
    # test_data1_gen = generate_dataset(test_data1_file, fileConfig, 'train', getDataSet, configs[0].extract_func, test_list)
    test_data2_gen  = generate_dataset(test_data2_file,fileConfig, 'train', getDataSet_parallel, 2, extract_func = configs[1].extract_func, data_list = test_list, num_processes=10)
    train_data1_gen  = generate_dataset(train_data1_file, fileConfig, 'train', getDataSet_parallel, 1, extract_func = configs[0].extract_func, data_list = train_list, num_processes=5)
    train_data2_gen  = generate_dataset(train_data2_file, fileConfig, 'train', getDataSet_parallel,2, extract_func = configs[1].extract_func, data_list = train_list,num_processes=10)

    # Load or generate datasets
    test_data1 = load_dataset(test_data1_file, fileConfig,'train')
    test_data2 = load_dataset(test_data2_file,fileConfig,'train')
    # valid_data1 = load_dataset(valid_data1_file, fileConfig)
    # valid_data2 = load_dataset(valid_data2_file, fileConfig)
    train_data1 = load_dataset(train_data1_file, fileConfig,'train')
    train_data2 = load_dataset(train_data2_file, fileConfig,'train')



    class SeededSampler(Sampler):
        def __init__(self, data_source, seed):
            self.data_source = data_source
            self.seed = seed

        def __iter__(self):
            g = torch.Generator()
            g.manual_seed(self.seed)
            indices = torch.randperm(len(self.data_source), generator=g).tolist()
            return iter(indices)

        def __len__(self):
            return len(self.data_source)

    # Assuming the existence of train_data1, train_data2, valid_data1, valid_data2
    seed_for_shuffle = 42  # Example seed

    train_loader1 = DataLoader(dataset=train_data1, batch_size=batch_size, sampler=SeededSampler(train_data1, seed_for_shuffle))
    train_loader2 = DataLoader(dataset=train_data2, batch_size=batch_size, sampler=SeededSampler(train_data2, seed_for_shuffle))
    # valid_loader1 = DataLoader(dataset=valid_data1, batch_size=batch_size, sampler=SeededSampler(valid_data1, seed_for_shuffle))
    # valid_loader2 = DataLoader(dataset=valid_data2, batch_size=batch_size, sampler=SeededSampler(valid_data2, seed_for_shuffle))
    # test_loader1 and test_loader2 are not shuffled, as per your request, so they are unchanged.

    test_loader1 = DataLoader(dataset=test_data1, batch_size=batch_size, shuffle=False)
    test_loader2 = DataLoader(dataset=test_data2, batch_size=batch_size, shuffle=False)


    a,b = test_data1[0]
    c,d = test_data2[0]
    print('wavelet shape:', a.shape)
    print('multiBand shape:', c.shape)
    configs[0].seq_len = len(a)
    configs[1].seq_len = len(c)
    configs[0].pred_len = len(a)
    configs[1].pred_len = len(c)

    ## Run auto-encoder

    train_info = (train_list, train_loader1, train_loader2)
    test_info = (test_list, test_loader1, test_loader2)
    model,history = mytrain(train_info, test_info , configs, fileConfig, device='cuda', num_epochs=300)

    def save_history(history, filename):
        """
        Save training history to a pickle file
        Args:
            history (dict): Training history dictionary
            filename (str): Base filename without extension
        """
        # Ensure the filename ends with .pkl
        if not filename.endswith('.pkl'):
            filename += '.pkl'
            
        with open(filename, 'wb') as f:
            pickle.dump(history, f)
        print(f"History saved to {filename}")
    save_history(history, './logs/history.pth')

