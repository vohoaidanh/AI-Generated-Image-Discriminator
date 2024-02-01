# -*- coding: utf-8 -*-
from torchvision import models

from models.model import CustomClassifier

def build_model(cfg):
    backbone = cfg.model.backbone
    num_classes = cfg.model.num_classes

    if backbone == 'resnet18':
        model = models.resnet18(weights='DEFAULT')
        model.fc = CustomClassifier(in_features=model.fc.in_features, num_classes=num_classes)
        
    return model

def load_model(cfg):
    model = build_model(cfg=cfg)

    return model