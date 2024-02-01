# -*- coding: utf-8 -*-

import torch
import torch.backends.cudnn as cudnn
from munch import munchify

cudnn.benchmark = True

from utils.func import load_config


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

def _metric(y_pre, y_gt):
    
    from sklearn.metrics import accuracy_score, f1_score, confusion_matrix
    acc = accuracy_score(y_gt, y_pre)
    f1 = f1_score(y_gt, y_pre, pos_label=CLASS_NAMES[0])  # Đặt pos_label là nhãn của lớp positive
    conf_matrix = confusion_matrix(y_gt, y_pre, labels=CLASS_NAMES)
    
    return {'accuracy': acc, 'f1_score': f1, 'conf_matrix': conf_matrix}

def evaluation(model, dataloaders):
    from tqdm import tqdm
    predict_list = []
    ground_truth = []
    for inputs, labels in tqdm(dataloaders):
        inputs = inputs.to(device)

        with torch.no_grad():
            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)
         
        predict_list += [CLASS_NAMES[i] for i in list(preds.cpu().detach().numpy())]
        ground_truth += [CLASS_NAMES[i] for i in list(labels.cpu().detach().numpy())]

    result = _metric(ground_truth, predict_list)
    
    print(result)
    return result

from models.model_builder import load_model
from data.builder import generate_dataset_from_folder
from torch.utils.data import DataLoader



if __name__ == '__main__':
    # Load config
    cfg = load_config()
    cfg = munchify(cfg)
    
    CLASS_NAMES = cfg.eval.classes
    CHECKPOINT = cfg.eval.checkpoint
    # Select device
    if cfg.base.device == 'auto':
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    else:
        device = cfg.base.device

    # Load checkpoint
    if device.type == 'cpu':
        state_dict = torch.load(CHECKPOINT, map_location=torch.device('cpu')) 
    else: 
        state_dict = torch.load(CHECKPOINT)
        
    # Load model
    model = load_model(cfg)
    model = model.to(device)
    model.load_state_dict(state_dict['model_state_dict'])
    model.eval()
    
    #Create dataset and dataloader for evaluation
    dataset = generate_dataset_from_folder(cfg, cfg.eval.data_path)
    dataloaders = DataLoader(dataset, batch_size=16, shuffle=True, num_workers=0)
    
    
   #Run evaluation
    evaluation(model, dataloaders)
    



    