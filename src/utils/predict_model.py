import torch
import torch.backends.cudnn as cudnn
import numpy as np
import pandas as pd
from config import CONFIG_DIR

cudnn.benchmark = True

from utils.load_config import load_config


config = load_config(CONFIG_DIR)
CLASS_NAMES = config['CLASSNAME']
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

def _eval(y_pre, y_gt):
    from sklearn.metrics import accuracy_score, f1_score, confusion_matrix

    acc = accuracy_score(y_gt, y_pre)
    f1 = f1_score(y_gt, y_pre, pos_label=CLASS_NAMES[0])  # Đặt pos_label là nhãn của lớp positive
    conf_matrix = confusion_matrix(y_gt, y_pre, labels=CLASS_NAMES)
    
    return {'accuracy': acc, 'f1_score': f1, 'conf_matrix': conf_matrix}

def predict(model, dataloaders, labels):
    predict_list = []
    for inputs, _ in dataloaders:
        inputs = inputs.to(device)

        with torch.no_grad():
            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)
         
        predict_list += [CLASS_NAMES[i] for i in list(preds.cpu().detach().numpy())]

    #result = np.concatenate((np.array(dataloaders[1]).reshape(-1, 1), np.array(predict_list).reshape(-1, 1)), axis=1)
    
    result = np.concatenate((np.array(labels).reshape(-1, 1), np.array(predict_list).reshape(-1, 1)), axis=1)
    df = pd.DataFrame(data=result, columns=['img_path','predict'])
    df.to_csv('../predict.csv', index=False)
    


def evaluation(model, dataloaders):
    predict_list = []
    ground_truth = []
    for inputs, labels in dataloaders:
        inputs = inputs.to(device)

        with torch.no_grad():
            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)
         
        predict_list += [CLASS_NAMES[i] for i in list(preds.cpu().detach().numpy())]
        ground_truth += [CLASS_NAMES[i] for i in list(labels.cpu().detach().numpy())]

    result = _eval(ground_truth, predict_list)
    print(result)
    return result
    









    