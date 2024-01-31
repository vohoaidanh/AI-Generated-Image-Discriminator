# -*- coding: utf-8 -*-
import os

BASE_DIR = os.path.abspath(os.path.dirname(__file__))
CONFIG_DIR = os.path.join(BASE_DIR, 'config.yaml')

#Train with LoRA
LORA = False
HUGGINGFACE_SOURCE = True
from peft import LoraConfig

LORA_CONFIG = LoraConfig(
        r=16,
        lora_alpha=16,
        target_modules=["conv1", "conv2"],
        lora_dropout=0.1,
        bias="none",
        modules_to_save=["classifier"],
    )