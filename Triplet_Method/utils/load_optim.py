import torch.optim as optim
import torch.optim.lr_scheduler as lr_scheduler

import torch.backends.cudnn as cudnn

cudnn.benchmark = True

import warnings
warnings.filterwarnings('ignore')

def load_optimization(cfg, model):
   
    OPTIM_FUNCTION = cfg.solver.optimizer if cfg.solver.optimizer else 'Adam'
    LEARNING_RATE = cfg.solver.learning_rate if cfg.solver.learning_rate else 0.00001

    try: 
        if OPTIM_FUNCTION=="Adam":
            optimizer_ft = optim.Adam(model.parameters(), lr=LEARNING_RATE)
        elif OPTIM_FUNCTION=="RAdam":
            optimizer_ft = optim.RAdam(model.parameters(), lr=LEARNING_RATE)
        elif OPTIM_FUNCTION=="SGD":
            optimizer_ft = optim.SGD(model.parameters(), lr=LEARNING_RATE, momentum=0.9)
        elif OPTIM_FUNCTION=="Adadelta":
            optimizer_ft = optim.Adadelta(model.parameters(), lr=LEARNING_RATE)
        elif OPTIM_FUNCTION=="Adagrad":
            optimizer_ft = optim.Adagrad(model.parameters(), lr=LEARNING_RATE)
        elif OPTIM_FUNCTION=="AdamW":
            optimizer_ft = optim.AdamW(model.parameters(), lr=LEARNING_RATE)
        elif OPTIM_FUNCTION=="Adamax":
            optimizer_ft = optim.Adamax(model.parameters(), lr=LEARNING_RATE)
        elif OPTIM_FUNCTION=="ASGD":
            optimizer_ft = optim.ASGD(model.parameters(), lr=LEARNING_RATE)
        elif OPTIM_FUNCTION=="NAdam":
            optimizer_ft = optim.NAdam(model.parameters(), lr=LEARNING_RATE)
        elif OPTIM_FUNCTION=="Rprop":
            optimizer_ft = optim.Rprop(model.parameters(), lr=LEARNING_RATE)

        # update config.

    except:
        print('Error: Could not find optim function. Please check optim function name.')
        exit(1)
        
    return optimizer_ft

def load_lr_scheduler(cfg, optimizer_ft):
    if cfg.solver.lr_scheduler=='StepLR':
        exp_lr_scheduler = lr_scheduler.StepLR(optimizer_ft, step_size=7, gamma=0.1)
    else:
        exp_lr_scheduler = lr_scheduler.StepLR(optimizer_ft, step_size=7, gamma=0.1)

    return exp_lr_scheduler
















