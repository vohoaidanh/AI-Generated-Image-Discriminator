# -*- coding: utf-8 -*-

from src.utils import load_config, load_data, train_model, load_loss_function, load_model, load_optimization, load_data_huggingface
import json

import torch
from torch.optim import lr_scheduler


import transformers
import accelerate
import peft

from config import CONFIG_DIR, LORA, LORA_CONFIG, HUGGINGFACE_SOURCE

print(f"Transformers version: {transformers.__version__}")
print(f"Accelerate version: {accelerate.__version__}")
print(f"PEFT version: {peft.__version__}")

def main():

    #model_checkpoint = "microsoft/resnet-18"
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    
    # load data
    if HUGGINGFACE_SOURCE:
        dataloaders, dataset_sizes, class_names = load_data_huggingface()
    else:
        dataloaders, dataset_sizes, class_names = load_data()
    # load model
    model = load_model()
    model = model.to(device)

    #Load and prepare a model
    def print_trainable_parameters(model):
        trainable_params = 0
        all_param = 0
        for _, param in model.named_parameters():
            all_param += param.numel()
            if param.requires_grad:
                trainable_params += param.numel()
        print(
            f"trainable params: {trainable_params} || all params: {all_param} || trainable%: {100 * trainable_params / all_param:.2f}"
        )
    
    print_trainable_parameters(model)
    
    from peft import get_peft_model
    
    if LORA:
        lora_model = get_peft_model(model, LORA_CONFIG)
        print_trainable_parameters(model = lora_model)
    else:
        lora_model = model
    # load loss function
    criterion = load_loss_function()
    
    # load optimization function
    optimizer_ft = load_optimization(model=lora_model)
    
    # Decay LR by a factor of 0.1 every 7 epochs
    exp_lr_scheduler = lr_scheduler.StepLR(optimizer_ft, step_size=7, gamma=0.1)
    
    
    print(json.dumps(load_config(CONFIG_DIR), 
                    sort_keys=True, indent=4))
    
    train_model(model=lora_model, device=device, dataloaders=dataloaders, 
                criterion=criterion, optimizer=optimizer_ft, scheduler=exp_lr_scheduler)


if __name__ == "__main__":
    
    main()
    
    