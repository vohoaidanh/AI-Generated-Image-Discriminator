import torch.nn as nn
import torch.backends.cudnn as cudnn

cudnn.benchmark = True
import warnings
warnings.filterwarnings('ignore')


def load_loss_function(cfg):
    # load config
    LOSS_FUNCTION = cfg.train.criterion if cfg.train.criterion else 'CrossEntropyLoss'

    try: 
        if LOSS_FUNCTION == "CrossEntropyLoss":
            criterion = nn.CrossEntropyLoss()
        elif LOSS_FUNCTION == "NLLLoss":
            criterion = nn.NLLLoss()
        
        # update config.

    except:
        print('Error: Could not find loss function. Please check loss function name.')
        exit(1)
    return criterion