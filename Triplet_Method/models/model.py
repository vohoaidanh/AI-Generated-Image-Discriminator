# -*- coding: utf-8 -*-


import torch.nn as nn

class CustomClassifier(nn.Module):
    def __init__(self, in_features=512, num_classes=2):
        super(CustomClassifier, self).__init__()
        self.fc = nn.Linear(in_features, num_classes)

    def forward(self, x):
        x = self.fc(x)
        return x

        