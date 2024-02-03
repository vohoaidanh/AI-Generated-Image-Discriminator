# -*- coding: utf-8 -*-
import os
import cv2
import torch
from torch.utils.data import Dataset
import numpy as np
import matplotlib.pyplot as plt
import json
from PIL import Image

class BaseDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        self.root_dir = root_dir
        self.classes = sorted(os.listdir(root_dir))
        self.class_to_idx = {cls_name: idx for idx, cls_name in enumerate(self.classes)}
        self.images = self._load_images()
        self.transform = transform
        self._len = self.__len__()

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img_name, class_name = self.images[idx]
        img_path = os.path.join(self.root_dir, class_name, img_name)
        #anchor_img = cv2.imread(img_path)  # Đọc hình ảnh bằng OpenCV
        anchor_img = Image.open(img_path)  # Đọc hình ảnh bằng Pil Image
        
        if self.transform:
            anchor_img = self.transform(anchor_img)

        return anchor_img, torch.tensor(self.class_to_idx[class_name])
    
    def _load_images(self):
        images = []
        for class_name in self.classes:
            class_path = os.path.join(self.root_dir, class_name)
            for img_name in os.listdir(class_path):
                images.append((img_name, class_name))
        return images
    
    def __repr__(self):
        
        dataset_info = {
            'classes': self.classes,
            'class_to_idx': self.class_to_idx,
            'len': self._len,
            'root_dir': self.root_dir
            }
        dataset_info = json.dumps(dataset_info, indent=4)

        return f'CustomDataset with info: {dataset_info}'



class TripletDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        self.root_dir = root_dir
        self.classes = sorted(os.listdir(root_dir))
        self.class_to_idx = {cls_name: idx for idx, cls_name in enumerate(self.classes)}
        self.images = self._load_images()
        self.transform = transform
        self._len = self.__len__()

    def __len__(self):
        return len(self.images)
    
    def __getitem__(self, idx):
        img_name, class_name = self.images[idx]
        img_path = os.path.join(self.root_dir, class_name, img_name)
        #image = Image.open(img_path).convert("RGB")
        anchor_img = cv2.imread(img_path)  # Đọc hình ảnh bằng OpenCV
        anchor_img = cv2.cvtColor(anchor_img, cv2.COLOR_BGR2RGB)
 
        while True:
            pos_idx = np.random.randint(0, self._len)
            pos_name, pos_class = self.images[pos_idx]
            if (pos_class == class_name) and (pos_idx != idx): break
        
        while True:
            neg_idx = np.random.randint(0, self._len)
            neg_name, neg_class = self.images[neg_idx]
            if (neg_class != class_name) and (neg_idx != idx): break
        
        pos_img = cv2.imread(os.path.join(self.root_dir, pos_class, pos_name))  # Đọc hình ảnh bằng OpenCV
        pos_img = cv2.cvtColor(pos_img, cv2.COLOR_BGR2RGB)
        
        neg_img = cv2.imread(os.path.join(self.root_dir, neg_class, neg_name))  # Đọc hình ảnh bằng OpenCV
        neg_img = cv2.cvtColor(neg_img, cv2.COLOR_BGR2RGB)
        
        # Chuyển đổi hình ảnh sang tensor
        #image = torch.from_numpy(image.transpose((2, 0, 1))).float()
        
        if self.transform:
            anchor_img = self.transform(anchor_img)
            pos_img = self.transform(pos_img)
            neg_img = self.transform(neg_img)
        
        # Return image and label (format at torch.tensor)
        return anchor_img, pos_img, neg_img, torch.tensor(self.class_to_idx[class_name])
    
    def _load_images(self):
        images = []
        for class_name in self.classes:
            class_path = os.path.join(self.root_dir, class_name)
            for img_name in os.listdir(class_path):
                images.append((img_name, class_name))
        return images


def visual(dataset):
    indices = np.random.choice(len(dataset), 9, replace=False)
    fig, axes = plt.subplots(3, 3, figsize=(10, 10))
    
    for i, ax in enumerate(axes.flat):
            # Lấy hình ảnh từ tập dữ liệu
            image,_,_, label = dataset[indices[i]]
            image = (image+np.pi)/(2*np.pi)
            # Hiển thị hình ảnh
            ax.imshow(image.permute(1, 2, 0))  # Chuyển vị chiều của hình ảnh (C, H, W) thành (H, W, C)
            ax.set_title(f"Label: {label}")
            ax.axis('off')
        

# =============================================================================
# a = dataset[np.random.randint(0,600)]
# print(a[3])
# flat_matrix = a[0].flatten()
# plt.hist(flat_matrix, bins=360, color='blue', alpha=0.7)
# plt.title('Histogram of Matrix')
# plt.xlabel('Pixel Value')
# plt.ylabel('Frequency')
# plt.grid(True)
# plt.show()
# =============================================================================



