# -*- coding: utf-8 -*-
import numpy as np
from PIL import Image, ImageFilter,ImageChops
from scipy.fft import dctn, idctn, fft2, ifft2, fftshift
import cv2

def gaussian_filter(image,radius=5):
  """
  Áp dụng bộ lọc Gaussian cho ảnh PIL.Image.

  Tham số:
    image: Ảnh PIL.Image.
    kernel_size: Kích thước kernel của bộ lọc Gaussian.
    sigma: Độ lệch chuẩn của bộ lọc Gaussian.

  Trả về:
    Ảnh PIL.Image sau khi áp dụng bộ lọc Gaussian.
  """

  # Chuyển đổi ảnh sang thang độ xám
  #image = image.convert("L")

  # Tạo kernel Gaussian
  kernel = ImageFilter.GaussianBlur(radius)
  r, g, b = image.split()
  r_filtered = r.filter(kernel)
  g_filtered = g.filter(kernel)
  b_filtered = b.filter(kernel)
  
  # Áp dụng bộ lọc Gaussian
  filtered_image = Image.merge("RGB", (r_filtered, g_filtered, b_filtered))

  # Chuyển đổi ảnh trở lại RGB
  #filtered_image = filtered_image.convert("RGB")

  return filtered_image

# Ví dụ sử dụng
image = Image.open("../images/real1.jpg")
image.show()
# Áp dụng bộ lọc Gaussian với kernel size 5 và sigma 2
filtered_image = gaussian_filter(image, 3)


sub_image = ImageChops.subtract(image, filtered_image)


# Hiển thị ảnh sau khi áp dụng bộ lọc
sub_image.show()


a = np.array(sub_image)
b = np.array(image)
c = np.array(filtered_image)
d = np.array(filtered_image)

image = np.array(image)
blur = cv2.GaussianBlur(image, (3,3),0)



import matplotlib.pyplot as plt

def _norm(image):
    return np.asarray((image-np.min(image))/(np.max(image)-np.min(image))*255.0, dtype='float32')

diff = image - blur

r = diff[:,:,0]
g = diff[:,:,1]
b = diff[:,:,2]


r = np.abs(fftshift(fft2(r)))
g = np.abs(fftshift(fft2(g)))
b = np.abs(fftshift(fft2(b)))

a = cv2.merge([r, g, b])

e = np.log(1+a)

e = _norm(e)

plt.imshow(_norm(np.abs(e[:,:,:])))

f = Image.fromarray(np.asarray(e,dtype=np.uint8))

plt.imshow(e)
























