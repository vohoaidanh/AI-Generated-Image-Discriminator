# -*- coding: utf-8 -*-
import cv2
import numpy as np
import torch
import torchvision.transforms as transforms
import torchvision.transforms.functional as TF
from typing import Tuple

from scipy.fft import dctn, idctn, fft2, ifft2, fftshift

from random import random, choice
from io import BytesIO
from PIL import Image, ImageFilter,ImageChops
from scipy.ndimage.filters import gaussian_filter

rz_dict = {'bilinear': Image.BILINEAR,
           'bicubic': Image.BICUBIC,
           'lanczos': Image.LANCZOS,
           'nearest': Image.NEAREST}


def __norm(image):
    return (image - np.min(image))/(np.max(image) - np.min(image))

def __resize_with_pad(image: np.array, 
                    dsize: Tuple[int, int], 
                    padding_color: Tuple[int] = (255, 255, 255)) -> np.array:
    """Maintains aspect ratio and resizes with padding.
    Params:
        image: Image to be resized.
        new_shape: Expected (width, height) of new image.
        padding_color: Tuple in BGR of padding color
    Returns:
        image: Resized image with padding
    """
    original_shape = (image.shape[1], image.shape[0])
    ratio = float(max(dsize))/max(original_shape)
    new_size = tuple([int(x*ratio) for x in original_shape])
    image = cv2.resize(image, new_size)
    delta_w = dsize[0] - new_size[0]
    delta_h = dsize[1] - new_size[1]
    top, bottom = delta_h//2, delta_h-(delta_h//2)
    left, right = delta_w//2, delta_w-(delta_w//2)
    image = cv2.copyMakeBorder(image, top, bottom, left, right, cv2.BORDER_CONSTANT, value=padding_color)
    
    return image

def __custom_resize(img, cfg):
    interp = __sample_discrete(cfg.data_augment.rz_interp)
    return TF.resize(img, cfg.data_augment.loadSize, interpolation=rz_dict[interp])

def __sample_continuous(s):
    if len(s) == 1:
        return s[0]
    if len(s) == 2:
        rg = s[1] - s[0]
        return random() * rg + s[0]
    raise ValueError("Length of iterable s should be 1 or 2.")
    
def __sample_discrete(s):
    if len(s) == 1:
        return s[0]
    return choice(s)

def __gaussian_blur(img, sigma):
    gaussian_filter(img[:,:,0], output=img[:,:,0], sigma=sigma)
    gaussian_filter(img[:,:,1], output=img[:,:,1], sigma=sigma)
    gaussian_filter(img[:,:,2], output=img[:,:,2], sigma=sigma)
    
def __cv2_jpg(img, compress_val):
    img_cv2 = img[:,:,::-1]
    encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), compress_val]
    result, encimg = cv2.imencode('.jpg', img_cv2, encode_param)
    decimg = cv2.imdecode(encimg, 1)
    return decimg[:,:,::-1]
    
def __pil_jpg(img, compress_val):
    out = BytesIO()
    img = Image.fromarray(img)
    img.save(out, format='jpeg', quality=compress_val)
    img = Image.open(out)
    # load from memory before ByteIO closes
    img = np.array(img)
    out.close()
    return img



jpeg_dict = {'cv2': __cv2_jpg, 'pil': __pil_jpg}
def __jpeg_from_key(img, compress_val, key):
    method = jpeg_dict[key]
    return method(img, compress_val)
    


    
def get_transform(cfg):
    img_size = cfg.data.input_size
    
    def preprocess(image):
        nonlocal img_size
        
        image = np.array(image)

        image = cv2.resize(image, dsize=img_size)
        
        # Tách các kênh màu
        red_channel = image[:, :, 0]
        green_channel = image[:, :, 1]
        blue_channel = image[:, :, 2]

        # Áp dụng DCT cho từng kênh màu
        red_dct = __norm(dctn(red_channel, norm='ortho'))
        green_dct = __norm(dctn(green_channel, norm='ortho'))
        blue_dct = __norm(dctn(blue_channel, norm='ortho'))
        
        dct = np.stack((red_dct, green_dct, blue_dct), axis=2)

        dct = dct.transpose(2,1,0)
        dct = dct.astype(np.float32)
        #image = __resize_with_pad(image, dsize=img_size)
        
        gradient_x = cv2.Sobel(image, cv2.CV_32F, 1, 0, ksize=3)
        gradient_y = cv2.Sobel(image, cv2.CV_32F, 0, 1, ksize=3)
        magnitude, angle = cv2.cartToPolar(gradient_x, gradient_y, angleInDegrees=False)
        
        magnitude = magnitude/np.max(magnitude)
        magnitude = magnitude.transpose(2,1,0)
        magnitude = torch.from_numpy(magnitude)

        angle = angle/np.max(angle)      
        angle = angle.transpose(2,1,0)
        angle = torch.from_numpy(angle)
        
        #concatenated = torch.cat((magnitude, angle), dim=0)

        return dct
    
    def data_augment(img, cfg):
        img = np.array(img)

        if random() < cfg.data_augment.blur_prob:
            sig = __sample_continuous(cfg.data_augment.blur_sig)
            __gaussian_blur(img, sig)

        if random() < cfg.data_augment.jpg_prob:
            method = __sample_discrete(cfg.data_augment.jpg_method)
            qual = __sample_discrete(cfg.data_augment.jpg_qual)
            img = __jpeg_from_key(img, qual, method)

        return Image.fromarray(img)
    
    def fft_argment(image, cfg):
        
        image = np.array(image)
        blur = cv2.GaussianBlur(image, (3,3),0)
        diff = image - blur

        r = np.abs(fftshift(fft2(diff[:,:,0])))
        g = np.abs(fftshift(fft2(diff[:,:,1])))
        b = np.abs(fftshift(fft2(diff[:,:,2])))
        
        log_img = cv2.merge([r, g, b])
        log_img = np.log(1+log_img)
        log_img = (log_img + 0.5)/0.5
        log_img = log_img.transpose(2,1,0)
        log_img = torch.from_numpy(log_img).float()
        return log_img
      
    data_transforms = {
        'train': transforms.Compose([
            transforms.Lambda(lambda x: __custom_resize(x, cfg)),
            transforms.Lambda(lambda x: fft_argment(x,cfg)),
            #transforms.RandomHorizontalFlip(),  # Lật ảnh ngẫu nhiên theo chiều ngang
            #transforms.ToTensor(),  # Chuyển đổi hình ảnh thành tensor
            #transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]),
       
        'test': transforms.Compose([
            transforms.Lambda(lambda x: __custom_resize(x, cfg)),
            transforms.Lambda(lambda x: fft_argment(x,cfg)),
            #transforms.ToTensor(),  # Chuyển đổi hình ảnh thành tensor
            #transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]),
        
        'val': transforms.Compose([
            transforms.Lambda(lambda x: __custom_resize(x, cfg)),
            transforms.Lambda(lambda x: fft_argment(x,cfg)),
            #transforms.RandomHorizontalFlip(),  # Lật ảnh ngẫu nhiên theo chiều ngang
            #transforms.ToTensor(),  # Chuyển đổi hình ảnh thành tensor
            #transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]),
        
    }
    
    return data_transforms
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    