import random
import torch
import torch.backends.cudnn as cudnn
from torch.utils.data import DataLoader, Subset
from torchvision import transforms, datasets
import os
import argparse
from config import BASE_DIR, CONFIG_DIR
#from PIL import Image

cudnn.benchmark = True

from utils.load_model import load_model
from utils.load_config import load_config
from utils.predict_model import predict, evaluation

config = load_config(CONFIG_DIR)
IMG_SIZE = config['DATA']['IMG_SIZE'] if config['DATA']['IMG_SIZE'] else (224, 224)
CLASS_NAMES = config['CLASSNAME']
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


# declare transforms for dataset
data_transforms =  transforms.Compose([
        transforms.Resize(IMG_SIZE), # resize anh
        transforms.RandomAdjustSharpness(5.0), #sharpen image
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

def get_parse():
    parser = argparse.ArgumentParser()
    parser.add_argument('--test_path', type=str, required=False, default=r'RealFakeDB_tiny/test')
    parser.add_argument('--batch_size', type=int, required=False, default=16)
    parser.add_argument('--number_sample', type=int, required=False, default=40)
    args = parser.parse_args()
    return args

if __name__ == '__main__':
    args = get_parse()

    # load model
    model = load_model()
    model = model.to(device)
    model.eval()
            
    image_paths = args.test_path
    if not os.path.exists(args.test_path):
        image_paths = os.path.join(BASE_DIR, args.test_path)
    
    dataset = datasets.ImageFolder(image_paths,data_transforms)
    
    number_sample = 0
    if args.number_sample==0:
        number_sample = len(dataset)
    else:
        number_sample = args.number_sample
    
    subset_indices = random.sample(range(len(dataset)), number_sample)
    subset_dataset = Subset(dataset, subset_indices)
    subset_labels = [os.path.basename(dataset.samples[i][0]) for i in subset_indices]

    dataloaders = (DataLoader(subset_dataset, batch_size=args.batch_size, shuffle=False))
    
    evaluation(model, dataloaders)
    predict(model, dataloaders, subset_labels)

    
    
