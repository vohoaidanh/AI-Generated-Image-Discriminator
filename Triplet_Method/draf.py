from utils.func import load_config, print_trainable_parameters
from munch import munchify
from data.builder import generate_dataset, initialize_dataloader
import numpy as np
import matplotlib.pyplot as plt

cfg = load_config()
cfg = munchify(cfg)

train, test, val = generate_dataset(cfg)

dataloader = initialize_dataloader(cfg,train_dataset=train, test_dataset=test, val_dataset=val)

dataloader[2].shuffle = False
dataiter = iter(dataloader[2])

sample = next(dataiter)

#sample = np.asarray(sample[0,:,:])

#sample = sample.transpose((2,1,0))

sample[1]
