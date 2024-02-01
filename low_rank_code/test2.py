# -*- coding: utf-8 -*-

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.image import imread
import os
import time
from PIL import Image, ImageFilter

import cv2

# Đọc hình ảnh
image = cv2.imread("image1.jpg", cv2.IMREAD_GRAYSCALE)/255.0

plt.imshow(image)

gradient_x = cv2.Sobel(image, cv2.CV_64F, 1, 0, ksize=3)
gradient_y = cv2.Sobel(image, cv2.CV_64F, 0, 1, ksize=3)

plt.imshow(gradient_x,cmap='gray', vmin=0, vmax=1)
plt.imshow(gradient_y,cmap='gray', vmin=0, vmax=1)

magnitude, angle = cv2.cartToPolar(gradient_x, gradient_y, angleInDegrees=True)
angle = angle/np.max(angle)
plt.imshow(angle,cmap='gray', vmin=0, vmax=1)

grad = np.sqrt(gradient_x*gradient_x + gradient_y*gradient_y)

an = np.arctan2(gradient_y,gradient_x)
plt.imshow(an+3.14,cmap='gray', vmin=0, vmax=6.28)


laplacian = cv2.Laplacian(image, cv2.CV_64F)
laplacian = np.uint8(np.absolute(laplacian))

flat_matrix = an.flatten()
plt.hist(laplacian, bins=10, color='blue', alpha=0.7)
plt.title('Histogram of Matrix')
plt.xlabel('Pixel Value')
plt.ylabel('Frequency')
plt.grid(True)
plt.show()

plt.imshow(laplacian,cmap='gray', vmin=0, vmax=1)











