import os
import torch
import torch.backends.cudnn as cudnn
from torchvision import transforms
from .load_config import load_config, save_config
from config import CONFIG_DIR

from datasets import load_dataset

cudnn.benchmark = True

import warnings
warnings.filterwarnings('ignore')   



# load config
if os.path.exists(CONFIG_DIR):
    config = load_config(CONFIG_DIR)
else:
    config = load_config('config.yaml')

IMG_SIZE = config['DATA']['IMG_SIZE'] if config['DATA']['IMG_SIZE'] else (224, 224)
DATA_DIR = config['DATA']['DATA_DIR'] if config['DATA']['DATA_DIR'] else '../data'
BATCHSIZE = config['DATA']['BATCHSIZES'] if config['DATA']['BATCHSIZES'] else 16
NUM_WORKERS = config['DATA']['NUM_WORKERS'] if config['DATA']['NUM_WORKERS'] else 4
DATA_DIR_HUG = config['DATA']['DATA_DIR_HUG'] if config['DATA']['DATA_DIR_HUG'] else None

    # declare transforms for dataset
data_transforms = {
    'train': transforms.Compose([
        transforms.Resize(IMG_SIZE), # resize anh
        transforms.RandomAdjustSharpness(5.0), #sharpen image
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]),
    'val': transforms.Compose([
        transforms.Resize(IMG_SIZE), # resize anh
        transforms.RandomAdjustSharpness(5.0), #sharpen image
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
}

def collate_fn(batch):
    features = torch.stack([item["image"] for item in batch])
    labels = torch.tensor([item["label"] for item in batch])
    return [features, labels]

def train_transforms(sample):
    sample["image"] = [data_transforms['train'](image.convert("RGB")) for image in sample["image"]]
    return sample

def val_transforms(sample):
    
    sample["image"] = [data_transforms['val'](image.convert("RGB")) for image in sample["image"]]
    return sample

def load_data_huggingface():

    image_datasets={'train':'', 'val':''}
    
    image_datasets['val'] = load_dataset(DATA_DIR_HUG, streaming=False, split="validation")
    image_datasets['train'] = load_dataset(DATA_DIR_HUG, streaming=False, split="train")
    
    image_datasets['val'].set_transform(train_transforms)
    image_datasets['train'].set_transform(val_transforms)
    
    dataloaders = {x: torch.utils.data.DataLoader(image_datasets[x], batch_size=BATCHSIZE,
                                                shuffle=True, num_workers=NUM_WORKERS, collate_fn=collate_fn)
                  for x in ['train', 'val']}

    dataset_sizes = {x: len(image_datasets[x]) for x in ['train', 'val']}
  
    # update config.yaml
    #config['CLASSNAME'] = class_names
    config['DATA']['IMG_SIZE'] = IMG_SIZE
    config['DATA']['DATA_DIR'] = DATA_DIR
    config['DATA']['BATCHSIZES'] = BATCHSIZE
    config['DATA']['NUM_WORKERS'] = NUM_WORKERS
    save_config(config, 'config.yaml')

    return dataloaders, dataset_sizes, config['CLASSNAME']

