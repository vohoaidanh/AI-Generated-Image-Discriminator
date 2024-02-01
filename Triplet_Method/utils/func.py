# -*- coding: utf-8 -*-
import yaml

def load_config(path='config/default.yaml'):
    with open(path, 'r') as file:
        cfg = yaml.load(file, Loader=yaml.FullLoader)
    return cfg