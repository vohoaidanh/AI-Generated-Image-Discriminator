import time
import os
from sklearn.metrics import accuracy_score, f1_score
import torch
import torch.backends.cudnn as cudnn
from tqdm import tqdm

from torch.utils.tensorboard import SummaryWriter


cudnn.benchmark = True

import warnings
warnings.filterwarnings('ignore')

def train_model(cfg, model, device, dataloaders, criterion, optimizer, scheduler):
  
    writer = SummaryWriter('logs')

    since = time.time()
    best_acc = 0.0

    # create dir save weights
    SAVE_WEIGHT_PATH = cfg.base.save_weight_path
    
    if not os.path.isdir(SAVE_WEIGHT_PATH): 
        os.makedirs(SAVE_WEIGHT_PATH)
    
    EPOCHS = cfg.train.epochs
    for epoch in range(EPOCHS):
        print(f'Epoch {epoch}/{EPOCHS - 1}\n{"-" * 10}')

        time_epoch = time.time()    # start time of phase

        # Each epoch has a training and validation phase
        for phase in ['train', 'val']:
            model.train() if phase == "train" else model.eval()

            running_loss = 0.0      #loss value of phase
            f1 = 0                  # f1 score of phase
            list_predict = []       # predict value
            list_groundtruth = []   # groundtruth value

            # Iterate over data.
            with tqdm(dataloaders[phase], desc=f'{phase.capitalize()} Batch', unit='batch', leave=False, position=0, mininterval=0.5) as batch_progress:
                
                for inputs, labels in batch_progress:

                    inputs = inputs.to(device)
                    labels = labels.to(device)
    
                    # zero the parameter gradients
                    optimizer.zero_grad()
    
                    # forward
                    # track history if only in train
                    with torch.set_grad_enabled(phase == 'train'):
                        outputs = model(inputs)
                        _, preds = torch.max(outputs, 1)
                        loss = criterion(outputs, labels)
    
                        # backward + optimize only if in training phase
                        if phase == 'train':
                            loss.backward()
                            optimizer.step()

                    # statistics
                    running_loss += loss.item() * inputs.size(0)
                    list_predict += list(preds.cpu().detach().numpy())
                    list_groundtruth += list(labels.data.cpu().detach().numpy())
                    
                    # Update the progress bar with the current batch 
                    batch_progress.set_postfix({'Epoch Loss': running_loss / len(list_groundtruth), 'F1 Score': f1_score(list_groundtruth, list_predict, average='macro')})

            epoch_loss = running_loss / len(list_groundtruth)
            writer.add_scalar(f'{phase}/loss', epoch_loss, epoch)
            
            f1 = f1_score(list_groundtruth, list_predict, average='macro')
            acc = accuracy_score(list_groundtruth, list_predict)
            writer.add_scalar(f'{phase}/accuracy', acc, epoch)

            if phase == 'train':
                scheduler.step()

            if phase == "train":
                print(f'In the training phase:\t Loss = {epoch_loss:.4f}\t Acc score: {acc:.4f}\t F1_score = {f1:.4f}')
            else: 
                print(f'In the validation phase: Loss = {epoch_loss:.4f}\t Acc score: {acc:.4f}\t F1_score = {f1:.4f}')

            # Save the best model

            if phase == 'val' and f1 > best_acc:
                best_acc = f1
                torch.save({
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict()
                    }, os.path.join(SAVE_WEIGHT_PATH, 'best.pt'))
            
        # save lasted model
        torch.save({
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict()
            }, os.path.join(SAVE_WEIGHT_PATH, 'latest.pt'))
            
        print(f'Total time for epoch {epoch}: {time.time()-time_epoch:.2f}s\n')
    writer.close()
    time_elapsed = time.time() - since
    print(f'\n\nTraining complete in {time_elapsed // 60:.0f}m {time_elapsed % 60:.0f}s')
    print(f'Best val Acc: {best_acc:4f}')

    # update config.
    # None
    











