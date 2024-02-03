# -*- coding: utf-8 -*-

import os
from torch.utils.data import DataLoader
from .dataset import TripletDataset, BaseDataset
from .preprocess import get_transform
import torchvision.transforms as transforms
import torch
import cv2
import numpy as np
from datasets import load_dataset

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
    # This function for generate dataset for evaluation
    data_transforms = get_transform(cfg)

    if os.path.exists(root_dir):
        dataset = BaseDataset(root_dir, transform = data_transforms['train'])
    else:
        raise ValueError("Not found {0}".format(root_dir))
        return None
    return dataset
    

def __collate_fn(batch):
    features = torch.stack([item["image"] for item in batch])
    labels = torch.tensor([item["label"] for item in batch])
    return [features, labels]


def initialize_dataloader_huggingface(cfg):
    
    data_transforms = get_transform(cfg)
    
    DATA_DIR_HUG = cfg.huggingface.dataset
    BATCHSIZE = cfg.train.batch_size
    NUM_WORKERS = cfg.train.num_workers
    SPLITS = cfg.huggingface.split
    #CLASSNAME = cfg.huggingface.classes
    
    def __train_transforms(sample):
        sample["image"] = [data_transforms['train'](image.convert("RGB")) for image in sample["image"]]
        return sample
    
    def __test_transforms(sample):
        sample["image"] = [data_transforms['test'](image.convert("RGB")) for image in sample["image"]]
        return sample

    def __val_transforms(sample):
        sample["image"] = [data_transforms['val'](image.convert("RGB")) for image in sample["image"]]
        return sample
    
    def __get_transforrm(split='train'):
        if split == 'train':
            return __train_transforms
        elif split == 'test':
            return __test_transforms
        else:
            return __val_transforms

    image_datasets={}
    for _split in SPLITS:  
        image_datasets[_split] = load_dataset(DATA_DIR_HUG, streaming=False, split=_split)
    
    for _split in SPLITS:
        _trans = __get_transforrm(_split)
        image_datasets[_split].set_transform(_trans)
    
    dataloaders = {x: torch.utils.data.DataLoader(image_datasets[x], batch_size=BATCHSIZE,
                                                shuffle=True, num_workers=NUM_WORKERS, collate_fn=__collate_fn)
                  for x in SPLITS}

    #dataset_sizes = {x: len(image_datasets[x]) for x in SPLITS}
  
    # update config.yaml

    return dataloaders

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
    train_loader, test_loader, val_loader = None, None, None
    
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
    










    