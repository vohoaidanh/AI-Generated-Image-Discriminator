# -*- coding: utf-8 -*-
import yaml

def load_config(path='config/default.yaml'):
    with open(path, 'r') as file:
        cfg = yaml.load(file, Loader=yaml.FullLoader)
    return cfg

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