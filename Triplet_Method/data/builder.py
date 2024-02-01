# -*- coding: utf-8 -*-

import os
from torch.utils.data import DataLoader
from .dataset import TripletDataset, BaseDataset
from .preprocess import get_transform
import torchvision.transforms as transforms
import torch
import cv2
import numpy as np

def generate_dataset(cfg):
    
    data_path = cfg.base.data_path
    train_path = os.path.join(data_path, 'train')
    test_path = os.path.join(data_path, 'test')
    val_path = os.path.join(data_path, 'val')
    
    data_transforms = get_transform(cfg)   

    train_dataset, test_dataset, val_dataset = None, None, None
    
    if os.path.exists(train_path):
        train_dataset = BaseDataset(train_path, transform=data_transforms['train'])
    
    if os.path.exists(test_path):
        test_dataset = BaseDataset(test_path, transform=data_transforms['train'])
        
    if os.path.exists(val_path):
        val_dataset = BaseDataset(val_path, transform=data_transforms['val'])
    
    return train_dataset, test_dataset, val_dataset

def generate_dataset_from_folder(cfg, root_dir):
    
    data_transforms = get_transform(cfg)

    if os.path.exists(root_dir):
        dataset = BaseDataset(root_dir, transform = data_transforms['train'])
    else:
        raise ValueError("Not found {0}".format(root_dir))
        return None
    return dataset
    
def generate_triplet_dataset(cfg):
        
    data_path = cfg.base.data_path
    img_size = cfg.data.input_size
    train_path = os.path.join(data_path, 'train')
    test_path = os.path.join(data_path, 'test')
    val_path = os.path.join(data_path, 'val')
    
    def custom_transfrom(image):
        nonlocal img_size
        image = cv2.resize(image, dsize=img_size)
        gradient_x = cv2.Sobel(image, cv2.CV_64F, 1, 0, ksize=3)
        gradient_y = cv2.Sobel(image, cv2.CV_64F, 0, 1, ksize=3)
        magnitude, angle = cv2.cartToPolar(gradient_x, gradient_y, angleInDegrees=True)
        angle = angle/np.max(angle)
        angle = angle.transpose(2,1,0)
        return torch.tensor(angle, dtype=torch.float32)
        
        
    data_transforms = {
        'train': transforms.Compose([
            transforms.Lambda(lambda x: custom_transfrom(x)),
        ]),
        'val': transforms.Compose([
            transforms.Lambda(lambda x: custom_transfrom(x)),
        ])
    }
    
   
    train_dataset = TripletDataset(train_path, transform=data_transforms['train'])
    test_dataset = TripletDataset(test_path, transform=data_transforms['train'])
    val_dataset = TripletDataset(val_path, transform=data_transforms['val'])
    
    return train_dataset, test_dataset, val_dataset

def initialize_dataloader(cfg, train_dataset=None, test_dataset=None, val_dataset=None):
    batch_size = cfg.train.batch_size
    num_workers = cfg.train.num_workers
    pin_memory = cfg.train.pin_memory
    if train_dataset:
        train_loader = DataLoader(
            train_dataset,
            batch_size=batch_size,
            shuffle=True,
            num_workers=num_workers,
            drop_last=True,
            pin_memory=pin_memory
        )
    
    if val_dataset:
        val_loader = DataLoader(
            val_dataset,
            batch_size=batch_size,
            shuffle=True,
            num_workers=num_workers,
            drop_last=False,
            pin_memory=pin_memory
        )
        
    if test_dataset:
        test_loader = DataLoader(
            test_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            drop_last=False,
            pin_memory=pin_memory
        )
        
    return train_loader, test_loader, val_loader
    










    