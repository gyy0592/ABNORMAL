

import torch
import torch.nn as nn

class myTestLoss(nn.Module):
    def __init__(self):
        super(myTestLoss, self).__init__()
        self.mse_loss = nn.MSELoss(reduction='none')

    def forward(self, input_list, target_list, subBands2remove=None):
        temp_loss = []
        final_loss = []
        
        for idx, (input, target) in enumerate(zip(input_list, target_list)): 
            # Check if it's the first element
            if input.shape[1] == 400 and subBands2remove is not None:
                # First transpose to get (batch_size, 15, 400)
                input = input.transpose(-2, -1)  # Now (batch_size, 15, 400)
                target = target.transpose(-2, -1)  # Now (batch_size, 15, 400)
                
                # Reshape to (batch_size, 300, 20) where 300 is channel
                input = input.reshape(-1, 300, 20)
                target = target.reshape(-1, 300, 20)
                
                # Apply band removal
                for band_idx in subBands2remove:
                    input[:, band_idx, :] = 0
                    input[:, band_idx + 100, :] = 0
                    input[:, band_idx + 200, :] = 0
                    
                    target[:, band_idx, :] = 0
                    target[:, band_idx + 100, :] = 0
                    target[:, band_idx + 200, :] = 0
                
                # Reshape back to (batch_size, 15, 400)
                input = input.reshape(-1, 15, 400)
                target = target.reshape(-1, 15, 400)
                
                # Transpose back to original dimension order (batch_size, 400, 15)
                input = input.transpose(-2, -1)
                target = target.transpose(-2, -1)
                
                # Calculate loss
                loss = torch.nn.MSELoss(reduction='none')(input, target).mean(dim=[1, 2]) * 100 / (100 - len(subBands2remove))
                temp_loss.append(loss)
            elif input.shape[1] == 400:
                # First transpose to get (batch_size, 15, 400)
                input = input.transpose(-2, -1)  # Now (batch_size, 15, 400)
                target = target.transpose(-2, -1)  # Now (batch_size, 15, 400)
                
                # Reshape to (batch_size, 300, 20) where 300 is channel
                input = input.reshape(-1, 300, 20)
                target = target.reshape(-1, 300, 20)
                
                # input[:, 200:, :] *=1e-2
                # target[:, 200:, :] *=1e-2
                
                # Reshape back to (batch_size, 15, 400)
                input = input.reshape(-1, 15, 400)
                target = target.reshape(-1, 15, 400)
                
                # Transpose back to original dimension order (batch_size, 400, 15)
                input = input.transpose(-2, -1)
                target = target.transpose(-2, -1)
                
                # Calculate loss
                loss = torch.nn.MSELoss(reduction='none')(input, target).mean(dim=[1, 2])
                temp_loss.append(loss)
                
                # For other elements, calculate loss normally
            elif input.shape[1] == 20:
                # First transpose to get (batch_size, 15, 20)
                input = input.transpose(-2, -1)  # Now (batch_size, 15, 20)
                target = target.transpose(-2, -1)  # Now (batch_size, 15, 20)
                
                # Reshape to (batch_size, 300, 20) where 300 is channel
                input = input.reshape(-1, 15, 20) 
                target = target.reshape(-1, 15, 20) 
                # apply factor
                input[:, 10:, :] *=1e-2
                target[:, 10:, :] *=1e-2
                
                input = input.reshape(-1, 15, 20)
                target = target.reshape(-1, 15, 20)
                
                input = input.transpose(-2, -1)
                target = target.transpose(-2, -1)
                
                # Calculate loss
                loss = torch.nn.MSELoss(reduction='none')(input, target).mean(dim=[1, 2])
                temp_loss.append(loss)
                
                # For other elements, calculate loss normally
            else:
                temp_loss.append(torch.nn.MSELoss(reduction='none')(input, target).mean(dim=[1, 2]))

        # Combine losses for each batch
        for lp in range(len(temp_loss[0])):
            final_loss.append(temp_loss[0][lp] + temp_loss[1][lp])

        return final_loss
    

class trainLoss(nn.Module):
    def __init__(self):
        super(trainLoss, self).__init__()

    def forward(self, input_list, target_list):
        # Shape for temp_losses is [batch_size, seq_length, n_features]
        # print(input.shape)
        num_path = len(input_list)
        final_loss = 0
        temp_loss = []
        for input,target in zip(input_list,target_list):
            # loss is mse loss
            if input.shape[1] == 400:
                input = input.transpose(-2, -1)
                target = target.transpose(-2, -1)
                input = input.reshape(-1, 300, 20)
                target = target.reshape(-1, 300, 20)
                # input[:, 200:, :] *=1e-2
                # target[:, 200:, :] *=1e-2
                input = input.reshape(-1, 15, 400)
                target = target.reshape(-1, 15, 400)
                input = input.transpose(-2, -1)
                target = target.transpose(-2, -1)
            elif input.shape[1] == 20:
                input = input.transpose(-2, -1)
                target = target.transpose(-2, -1)
                input = input.reshape(-1, 15, 20)
                target = target.reshape(-1, 15, 20)
                input[:, 10:, :] *=1e-2
                target[:, 10:, :] *=1e-2
                input = input.reshape(-1, 15, 20)
                target = target.reshape(-1, 15, 20)
                input = input.transpose(-2, -1)
                target = target.transpose(-2, -1)

            # final_loss+= torch.nn.MSELoss(input, target)
            temp_losses = torch.nn.MSELoss(reduction='none')(input, target)
            mean_losses = temp_losses.mean()
            # final_loss += max_losses.mean()
            temp_loss.append(mean_losses)
        # temp_loss[1]*=1e1
        # ratio1 = temp_loss[0]/temp_loss[1]
        # ratio2 = temp_loss[1]/temp_loss[0]
        # final_loss = temp_loss[0]*100+temp_loss[1]
        final_loss = temp_loss[0]+temp_loss[1]
        # final_loss = max(temp_loss[0],temp_loss[1])
        # final_loss = torch.stack(temp_loss).mean()
        # Final loss should be 0.7 of larger one plus 0.3 of smaller one, first decide which one is larger

        
        return final_loss
