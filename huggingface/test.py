# -*- coding: utf-8 -*-

from src.utils import *
import json

import torch
from torch.optim import lr_scheduler

import transformers
import accelerate
import peft

print(f"Transformers version: {transformers.__version__}")
print(f"Accelerate version: {accelerate.__version__}")
print(f"PEFT version: {peft.__version__}")

model_checkpoint = "microsoft/resnet-18"
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# load data
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


from peft import LoraConfig, get_peft_model

config = LoraConfig(
    r=1,
    lora_alpha=16,
    target_modules=["conv1"],
    lora_dropout=0.1,
    bias="none",
    modules_to_save=["classifier"],
)

lora_model = get_peft_model(model, config)
print_trainable_parameters(model = lora_model)

# load loss function
criterion = load_loss_function()

# load optimization function
optimizer_ft = load_optimization(model=lora_model)

# Decay LR by a factor of 0.1 every 7 epochs
exp_lr_scheduler = lr_scheduler.StepLR(optimizer_ft, step_size=7, gamma=0.1)


print(json.dumps(load_config(r'config.yaml'), 
                sort_keys=True, indent=4))

train_model(model=lora_model, device=device, dataloaders=dataloaders, 
            criterion=criterion, optimizer=optimizer_ft, scheduler=exp_lr_scheduler)


model.layer4[1].conv1.base_layer



model_state_dict = model.state_dict()










