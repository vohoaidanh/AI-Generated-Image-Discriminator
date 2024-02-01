# -*- coding: utf-8 -*-


import numpy as np
import matplotlib.pyplot as plt
from matplotlib.image import imread
import os
import time

np.random.seed(0)

a,b,m = 4,5,6

I = np.random.randn(a,b)
W_q = np.random.randn(b,m)
W_k = np.random.randn(b,m)

k1 = I @ W_q
k2 = I @ W_k
k3 = k1 @ k2.T

_q = W_q @ W_k.T

k4 = I @ _q @ I.T

print(np.allclose(k3, k4))

from transformers import AutoModel
vit_model = AutoModel.from_pretrained('google/vit-base-patch16-224')

state_dict_vit = vit_model.state_dict()

state_dict_vit['embeddings.patch_embeddings.projection.bias'].size()


from vit import ViTForClassfication
import torch

device = "cuda" if torch.cuda.is_available() else "cpu"

config = {
    "patch_size": 4,  # Input image size: 32x32 -> 8x8 patches
    "hidden_size": 48,
    "num_hidden_layers": 4,
    "num_attention_heads": 4,
    "intermediate_size": 4 * 48, # 4 * hidden_size
    "hidden_dropout_prob": 0.0,
    "attention_probs_dropout_prob": 0.0,
    "initializer_range": 0.02,
    "image_size": 32,
    "num_classes": 10, # num_classes of CIFAR10
    "num_channels": 3,
    "qkv_bias": True,
    "use_faster_attention": False,
}

# These are not hard constraints, but are used to prevent misconfigurations
assert config["hidden_size"] % config["num_attention_heads"] == 0
assert config['intermediate_size'] == 4 * config['hidden_size']
assert config['image_size'] % config['patch_size'] == 0


vit_custom_model = ViTForClassfication(config)


from torch import nn

query = nn.Linear(48, 12, bias=True)
d = query.state_dict()
d = d['weight']
# =============================================================================
# 
# import torch
# from torchvision import models
# from src.utils import load_config, load_data, train_model, load_loss_function, load_model, load_optimization, load_data_huggingface
# from peft import get_peft_model, LoraConfig
# from transformers import AutoModel
# 
# 
# 
# model = models.resnet18()
# status_dict = model.state_dict()
# 
# 
# device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
# lora_model = load_model()
# 
# status_dict_lora = lora_model.state_dict()
# 
# 
# 
# 
# vit_model = AutoModel.from_pretrained('google/vit-base-patch16-224')
# 
# status_dict_vit = vit_model.state_dict()
# 
# 
# vit_config = LoraConfig(
#         r=16,
#         lora_alpha=16,
#         target_modules=["query", "value"],
#         lora_dropout=0.1,
#         bias="none",
#         modules_to_save=["classifier"],
#     )
# 
# lora_vit_model = get_peft_model(vit_model, vit_config)
# 
# 
# vit_config2 = LoraConfig(
#         r=4,
#         lora_alpha=4,
#         target_modules=["projection"],
#         lora_dropout=0.1,
#         bias="none",
#         modules_to_save=["classifier"],
#     )
# 
# 
# lora_vit_model = get_peft_model(vit_model, vit_config2)
# 
# status_dict_lora_vit = lora_vit_model.state_dict()
# 
# =============================================================================
