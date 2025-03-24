

import os
import torch
import errno



def load_dataset(file_name, fileConfig, type):
    if type == 'train':
        full_path = os.path.join(fileConfig.temp_training, file_name)
    elif type == 'test':
        full_path = os.path.join(fileConfig.temp_testing, file_name)
    elif type =='temp':
        full_path = os.path.join('/home/guoyiyang/github_repo/OoD_nonStationary/processed_data/temp' , file_name)
    # Load dataset
    print(f"loaded path {full_path}")
    dataset = torch.load(full_path)
    return dataset

def generate_dataset(file_name, fileConfig, type, generation_function, reshape_idx=0, **args):
    if type == 'train':
        full_path = os.path.join(fileConfig.temp_training, file_name)
    elif type == 'test':
        full_path = os.path.join(fileConfig.temp_testing, file_name)
    elif type =='temp':
        full_path = os.path.join('/home/guoyiyang/github_repo/OoD_nonStationary/processed_data/temp' , file_name)

    base, ext = os.path.splitext(full_path)
    # Check the existence of the temp file
    if os.path.isfile(full_path):
        print("Temp file exists, loading temp file")

        return 0

    # Generate dataset
    # print('saving files')
    # First check whether folder exists if not create the folder
    if not os.path.exists(os.path.dirname(full_path)):
        try:
            os.makedirs(os.path.dirname(full_path))
        except OSError as exc:
            if exc.errno != errno.EEXIST:
                raise  
    print(f"Generating dataset {full_path}")       
    dataset = generation_function(**args)
    torch.save(dataset, full_path)
    return 0