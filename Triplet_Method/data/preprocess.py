# -*- coding: utf-8 -*-
import cv2
import numpy as np
import torch
import torchvision.transforms as transforms
   
    


def get_transform(cfg):
    img_size = cfg.data.input_size
    
    def preprocess(image):
        nonlocal img_size
        image = cv2.resize(image, dsize=img_size)
        gradient_x = cv2.Sobel(image, cv2.CV_32F, 1, 0, ksize=3)
        gradient_y = cv2.Sobel(image, cv2.CV_32F, 0, 1, ksize=3)
        magnitude, angle = cv2.cartToPolar(gradient_x, gradient_y, angleInDegrees=False)
        
        magnitude = magnitude/np.max(magnitude)
        magnitude = magnitude.transpose(2,1,0)
        magnitude = torch.from_numpy(magnitude)

        angle = angle/np.max(angle)      
        angle = angle.transpose(2,1,0)
        angle = torch.from_numpy(angle)
        
        concatenated = torch.cat((magnitude, angle), dim=0)

        return angle
    
    data_transforms = {
        'train': transforms.Compose([
            transforms.Lambda(lambda x: preprocess(x)),
        ]),
       
        'test': transforms.Compose([
            transforms.Lambda(lambda x: preprocess(x)),
        ]),
        
        'val': transforms.Compose([
            transforms.Lambda(lambda x: preprocess(x)),
        ]),
        
    }
    
    return data_transforms
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    