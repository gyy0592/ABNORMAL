

import numpy as np
import torch
import re
from .BarryStatDetDataset import myloadNPZ
from ..myclass.loss_class import myTestLoss




def calculate_loss_and_snr(test_loader1, test_loader2, test_list, model, device, batch_size=1024, subBands2remove = None):
    model.to(device)
    model.eval()
    goodLoss = []
    badLoss = []
    good_indices = []
    bad_indices = []
    SNR = []
    classNums = []

    with torch.no_grad():
        global_index = 0

        for ((sequences1, x_mark_encs1), (sequences2, x_mark_encs2)) in zip(test_loader1, test_loader2):
            sequences = [sequences1.to(device), sequences2.to(device)]
            x_mark_encs = [x_mark_encs1.to(device), x_mark_encs2.to(device)]
            seq_preds = model(sequences, x_mark_encs)
            # Calculate losses for each instance in the batch using testLossFunc
            if subBands2remove is None:
                losses = myTestLoss()(seq_preds, sequences)
            else:
                losses = myTestLoss()(seq_preds, sequences, torch.tensor(subBands2remove).to(device))
            
            for loss in losses:
                test_loss = loss.item()

                if 'nonStationary' in test_list[global_index]:
                    badLoss.append(test_loss)
                    bad_indices.append(global_index)
                    SNR.append(myloadNPZ(test_list[global_index], 'SNR'))

                    extracted_number_match = re.search(r"classnum=(\d+)_", test_list[global_index])
                    extracted_number = extracted_number_match.group(1) if extracted_number_match else None
                    if extracted_number is None:
                        # extracted_number = 1
                        # raise error if classnum is not found
                        raise ValueError(f"Class number not found in {test_list[global_index]}")
                    classNums.append(int(extracted_number))
                else:
                    goodLoss.append(test_loss)
                    good_indices.append(global_index)

                global_index += 1

    return np.array(goodLoss), np.array(badLoss), np.array(SNR), np.array(classNums),(np.array(good_indices),np.array(bad_indices))