# -*- coding: utf-8 -*-

import os
from PIL import Image

def is_single_channel_image(image_path):
    try:
        # Mở ảnh bằng Pillow
        image = Image.open(image_path)

        # Kiểm tra số lượng kênh màu
        num_channels = len(image.getbands())

        # Đóng ảnh
        image.close()

        # Trả về True nếu chỉ có một kênh màu
        return num_channels == 1

    except Exception as e:
        # Xử lý các lỗi có thể xảy ra
        #print(f"Error: {e}")
        return False
    
from tqdm import tqdm

def find_single_channel_images(folder_path):
    single_channel_images = []

    # Duyệt qua tất cả các thư mục và tệp trong thư mục
    for root, dirs, files in os.walk(folder_path):
        for file in tqdm(files, desc="Checking files"):
            # Kiểm tra chỉ có một kênh màu
            file_path = os.path.join(root, file)
            if is_single_channel_image(file_path):
                single_channel_images.append(file_path)

    return single_channel_images

# Thay đổi đường dẫn của thư mục để kiểm tra
folder_path = "RealFakeDB_tiny"
single_channel_images = find_single_channel_images(folder_path)

#Delete
for i in single_channel_images:
    im = Image.open(i)
    im3 = im.convert("RGB")
    im3.save(i)
    print(i)



