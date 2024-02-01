# -*- coding: utf-8 -*-

import numpy as np
import cv2
import matplotlib.pyplot as plt


def transfrom(image):
    image = cv2.resize(image, dsize=(224,224))
    gradient_x = cv2.Sobel(image, cv2.CV_32F, 1, 0, ksize=3)
    gradient_y = cv2.Sobel(image, cv2.CV_32F, 0, 1, ksize=3)
    magnitude, angle = cv2.cartToPolar(gradient_x, gradient_y, angleInDegrees=True)
    angle = angle/np.max(angle)
    #angle = angle.transpose(2,1,0)
    return angle

img_path_list = []
angles = []
for img_path in img_path_list:
    image = cv2.imread(img_path,cv2.COLOR_BGR2RGB)
    angle = transfrom(image)
    angles.append(angle)

# Read the image
image = cv2.imread('images/real1.jpg',cv2.COLOR_BGR2RGB)
image = transfrom(image)
# Perform Fourier transform
f_transform = np.fft.fft2(image)

# Shift zero frequency component to the center
f_transform_shifted = np.fft.fftshift(f_transform)

# Calculate magnitude spectrum
magnitude_spectrum = np.log(np.abs(f_transform_shifted))
magnitude_spectrum = (magnitude_spectrum - np.min(magnitude_spectrum)) * 1 / (np.max(magnitude_spectrum)-np.min(magnitude_spectrum))
# Display original and Fourier transformed images
fig, axes = plt.subplots(1, 2, figsize=(20, 40))

plt.subplot(121), plt.imshow(image, cmap='gray')
plt.title('Input Image'), plt.xticks([]), plt.yticks([])
plt.subplot(122), plt.imshow(magnitude_spectrum, cmap='gray')
plt.title('Magnitude Spectrum'), plt.xticks([]), plt.yticks([])
plt.show()
