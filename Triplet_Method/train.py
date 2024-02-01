# -*- coding: utf-8 -*-

from utils.func import load_config, print_trainable_parameters
from data.builder import initialize_dataloader, generate_dataset, initialize_dataloader_huggingface
from models.model_builder import build_model
from munch import munchify

#----------
import torch
import torch.backends.cudnn as cudnn
from utils.load_optim import load_optimization, load_lr_scheduler
from utils.load_loss import load_loss_function

from models.trainer import train_model

import warnings
warnings.filterwarnings('ignore')

cudnn.benchmark = True

if __name__=="__main__":
    cfg = load_config()
    cfg = munchify(cfg)
    
    if cfg.base.device == 'auto':
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    else:
        device = cfg.base.device
        
    print('Training on ', device)

    #load data
    if cfg.huggingface.dataset:
        dataloaders = initialize_dataloader_huggingface(cfg)
    else:    
        train_dataset, test_dataset, val_dataset = generate_dataset(cfg)
        train_loader, test_loader, val_loader = initialize_dataloader(cfg,train_dataset, test_dataset, val_dataset)
        
        dataloaders = {
            'train': train_loader,
            'test': test_loader,
            'val': val_loader
            }
    
    
    # load model
    model = build_model(cfg)
    model = model.to(device)

    # load loss function
    criterion = load_loss_function(cfg)

    # load optimization function
    optimizer_ft = load_optimization(cfg=cfg,model=model)
    
    # Decay LR by a factor of 0.1 every 7 epochs
    exp_lr_scheduler = load_lr_scheduler(cfg=cfg, optimizer_ft=optimizer_ft)

    # train model
    train_model(cfg, model=model, device=device, dataloaders=dataloaders, 
                criterion=criterion, optimizer=optimizer_ft, scheduler=exp_lr_scheduler)
    

# =============================================================================
# import imutils
# import cv2
# 
# im = cv2.imread(r'images/real1.jpg')
# 
# im_224 = imutils.resize(im,width=224, height=224)
# 
# import matplotlib.pyplot as plt
# 
# plt.imshow(im_224)
# 
# 
# =============================================================================



