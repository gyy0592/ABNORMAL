import torch
import torch.optim as optim
import numpy as np 
from sklearn.metrics import roc_auc_score
import copy
from ..myclass.loss_class import trainLoss, myTestLoss
from ..myclass.mymodel import SignalReconstructionModel
import os 

def mytrain(train_info, test_info, model_config, fileConfig, device, num_epochs=100, scheduler_start_epoch=20, testLossFunc=myTestLoss(), criterion=trainLoss()):
    # delete all the files in fileConfig.weights_dir
    for file in os.listdir(fileConfig.weights_dir):
        os.remove(os.path.join(fileConfig.weights_dir, file))
    train_list, train_loader1, train_loader2 = train_info
    test_list, test_loader1, test_loader2 = test_info

    # Initialize model with provided configuration
    configs = model_config
    model = SignalReconstructionModel(configs, 2)
    model = model.to(device)

    # Initial learning rate
    initial_lr = 1e-4

    # Define the optimizer
    optimizer = optim.Adam(model.parameters(), lr=initial_lr, weight_decay=0)
    
    # def custom_lr_schedule(epoch):
    #     if epoch < 140:
    #         return optimizer.param_groups[0]['lr']
    #     elif epoch < 300:
    #         # Linear decay from 1e-4 to 1e-5 between epochs 140-300
    #         progress = (epoch - 140) / (300 - 140)
    #         return 1e-4 - (9e-5 * progress)
    #     else:
    #         return 1e-5

    # scheduler = LambdaLR(optimizer, custom_lr_schedule)

    # Placeholder for the best model's loss and weights
    best_stationary_loss = float('inf')
    best_model_wts = None

    # Path for saving best model weights
    best_model_path = fileConfig.weights_dir + 'best_weights.pth'
    print('weights at:', best_model_path)

    # Initialize history to store all metrics
    history = {
        'train_loss': [],
        'overall_auc': [],
        'glitch_aucs': [],
        'chirp_aucs': [],
        'class_specific_aucs': {},
        'learning_rates': [],
        'stationary_loss': [],
        'smoothed_val_loss': []  # NEW: For tracking smoothed validation loss
    }

    # NEW: Plateau-aware Early Stopping
    class PlateauStopper:
        def __init__(self, patience=50, threshold=0.001):
            self.best_loss = float('inf')
            self.patience = patience
            self.threshold = threshold
            self.counter = 0
            
        def step(self, current_loss):
            if current_loss < self.best_loss * (1 - self.threshold):
                self.best_loss = current_loss
                self.counter = 0
            else:
                self.counter += 1
            return self.counter >= self.patience

    plateau_stopper = PlateauStopper(patience=50)
    val_loss_window = []  # NEW: For validation loss smoothing

    for epoch in range(num_epochs):
        # Learning rate management
        if epoch < 140:
            if (epoch % 70) < 20:
                for param_group in optimizer.param_groups:
                    param_group['lr'] = 5*1e-4 
                model.unfreeze_path(0)
                model.freeze_path(1)
            else:
                for param_group in optimizer.param_groups:
                    param_group['lr'] = 1e-4 
                model.freeze_path(0)
                model.unfreeze_path(1)
        else:
            for param_group in optimizer.param_groups:
                param_group['lr'] = 1e-4
            model.unfreeze_path(0)
            model.unfreeze_path(1)
    
        model.eval()
        good_losses, bad_losses = [], []
        glitch_losses, chirp_losses = [], []
        class_specific_losses = {}
        stationary_losses = []
        test_list_idx = 0
        
        with torch.no_grad():
            for (sequence1, x_mark_enc1), (sequence2, x_mark_enc2) in zip(test_loader1, test_loader2):
                sequence = [sequence1.to(device), sequence2.to(device)]
                x_mark_enc = [x_mark_enc1.to(device), x_mark_enc2.to(device)]
                
                seq_pred = model(sequence, x_mark_enc)
                losses = testLossFunc(seq_pred, sequence)
                
                for loss in losses:
                    loss_value = loss.item()
                    filename = test_list[test_list_idx]

                    if 'stationaryNoise' in filename:
                        good_losses.append(loss_value)
                        stationary_losses.append(loss_value)
                    else:
                        bad_losses.append(loss_value)
                        if 'classnum=4' in filename or 'classnum=5' in filename:
                            chirp_losses.append(loss_value)
                        else:
                            glitch_losses.append(loss_value)
                        
                        if 'classnum=' in filename:
                            try:
                                class_num = int(filename.split('classnum=')[1].split('_')[0])
                                if class_num not in class_specific_losses:
                                    class_specific_losses[class_num] = []
                                class_specific_losses[class_num].append(loss_value)
                            except:
                                pass
                    
                    test_list_idx += 1

        model.train()
        train_loss = 0.0
        for (sequence1, x_mark_enc1), (sequence2, x_mark_enc2) in zip(train_loader1, train_loader2):
            sequence = [sequence1.to(device), sequence2.to(device)]
            x_mark_enc = [x_mark_enc1.to(device), x_mark_enc2.to(device)]

            seq_pred = model(sequence, x_mark_enc)
            loss = criterion(seq_pred, sequence)
            
            optimizer.zero_grad()
            loss.backward()
            
            # NEW: Gradient Clipping
            torch.nn.utils.clip_grad_norm_(
                model.parameters(), 
                max_norm=1.0,
                error_if_nonfinite=False
            )
            
            optimizer.step()
            train_loss += loss.item()

        avg_train_loss = train_loss / (len(train_loader1) + len(train_loader2))
        avg_stationary_loss = sum(stationary_losses) / len(stationary_losses) if stationary_losses else float('inf')

        # NEW: Validation Loss Smoothing
        val_loss_window.append(avg_stationary_loss)
        window_size = 10  # or 20
        if len(val_loss_window) > window_size:
            val_loss_window.pop(0)
        smoothed_val = np.mean(val_loss_window)
        history['smoothed_val_loss'].append(smoothed_val)

        # Calculate various AUCs
        try:
            overall_auc = roc_auc_score([0] * len(good_losses) + [1] * len(bad_losses), 
                                      good_losses + bad_losses)
        except ValueError as e:
            print(f"Error calculating overall_auc: {e}")
            overall_auc = 0

        try:
            glitch_auc = roc_auc_score([0] * len(good_losses) + [1] * len(glitch_losses), 
                                      good_losses + glitch_losses)
        except ValueError as e:
            print(f"Error calculating glitch_auc: {e}")
            glitch_auc = 0

        try:
            chirp_auc = roc_auc_score([0] * len(good_losses) + [1] * len(chirp_losses), 
                                     good_losses + chirp_losses)
        except ValueError as e:
            print(f"Error calculating chirp_auc: {e}")
            chirp_auc = 0

        epoch_class_aucs = {}
        for class_num, losses in class_specific_losses.items():
            try:
                class_auc = roc_auc_score([0] * len(good_losses) + [1] * len(losses), 
                                        good_losses + losses)
                epoch_class_aucs[class_num] = class_auc
            except ValueError as e:
                print(f"Error calculating AUC for class {class_num}: {e}")
                epoch_class_aucs[class_num] = 0

        # Store metrics in history
        history['train_loss'].append(avg_train_loss)
        history['overall_auc'].append(overall_auc)
        history['glitch_aucs'].append(glitch_auc)
        history['chirp_aucs'].append(chirp_auc)
        history['learning_rates'].append(optimizer.param_groups[0]['lr'])
        history['stationary_loss'].append(avg_stationary_loss)
        
        for class_num, auc in epoch_class_aucs.items():
            if class_num not in history['class_specific_aucs']:
                history['class_specific_aucs'][class_num] = []
            history['class_specific_aucs'][class_num].append(auc)
        
        print(f"Epoch {epoch + 1}/{num_epochs}")
        print(f"Training Loss: {avg_train_loss:.4e}")
        print(f"Stationary Test Loss: {avg_stationary_loss:.4e}")
        print(f"Smoothed Validation Loss: {smoothed_val:.4e}")  # NEW
        print(f"Overall AUC: {overall_auc:.4e}")
        print(f"Glitch AUC: {glitch_auc:.4e}")
        print(f"Chirp AUC: {chirp_auc:.4e}")
        print(f"Learning Rate: {optimizer.param_groups[0]['lr']}")
        for class_num, auc in epoch_class_aucs.items():
            print(f"Class {class_num} AUC: {auc:.4e}")

        # NEW: Check for early stopping
        if epoch > 140:
            if plateau_stopper.step(smoothed_val):
                print(f"Stopping early at epoch {epoch} due to plateau")
                break

        # scheduler.step()

        if avg_stationary_loss < best_stationary_loss and epoch > 20:
            best_stationary_loss = avg_stationary_loss
            best_model_wts = copy.deepcopy(model.state_dict())
            torch.save(model.state_dict(), best_model_path)
            print(f"Saved new best model with Stationary Loss: {avg_stationary_loss:.4e}")
        
        if epoch > 50 and abs(best_stationary_loss - avg_stationary_loss) < 0.01:
            torch.save(model.state_dict(), fileConfig.weights_dir + f'weights_epoch{epoch}.pth')

    # Load best model weights
    model.load_state_dict(best_model_wts)

    return model, history