# -*- coding: utf-8 -*-
import os
import pathlib
from PIL import Image
from tqdm import tqdm

def resize(image_path,output_path,size=224):
    """
        Thay đổi kích thước và lưu ảnh sang kích thước 224 (ở chiều nhỏ hơn)
        
        Args:
          image_path: Đường dẫn đến ảnh đầu vào
          output_path: Đường dẫn đến ảnh đầu ra
          width: Chiều rộng mong muốn của ảnh đầu ra
        
        Returns:
      None
    """
    
    
    
    try:
        image = Image.open(image_path)
    except IOError:
        # Ảnh không được mở thành công
        return

    w, h = image.size
    scale = size/min(w,h)
    new_w = int(w*scale)
    new_h = int(h*scale)
        
    resize_image = image.resize((new_w, new_h), resample=Image.LANCZOS)
    
    resize_image.save(output_path)


DATA_SOURCE = r'E:\RealFakeDB'
DATA_DESTINATION = r'E:\RealFakeDB_resize'

if __name__ == '__main__':
    
    list_set = ['train']
    list_class = ['real']
    file_ext = ['.webp','.jpg','.png','.jpeg']
    for s in list_set:
        for c in list_class:
            print(s+'/'+c)
  
            _file_paths = list(os.listdir(os.path.join(DATA_SOURCE, s, c)))
            file_paths = [i for i in _file_paths if pathlib.Path(i).suffix in file_ext]
            print('file_paths', file_paths[:10])
            
            t = tqdm(total=len(file_paths), desc="Processing items", unit="item")
            for image_path in file_paths:
                output_dir = os.path.join(DATA_DESTINATION, s, c)
                if not os.path.isdir(output_dir):
                    os.makedirs(output_dir, exist_ok=True)
                
                if os.path.isfile(os.path.join(DATA_DESTINATION, s, c, image_path)):
                    continue
                
                image_path_full = os.path.join(DATA_SOURCE, s, c, image_path)
                
                file_ext = pathlib.Path(image_path).suffix
                
                output_file = image_path.replace(file_ext, '.png')
                
                output_path_full = os.path.join(output_dir, output_file)
                                   
                resize(image_path_full, 
                       output_path = output_path_full,size=224)
                t.update()
            

    
    
    
    
    
    
    
    
    
    
    
    
    
    