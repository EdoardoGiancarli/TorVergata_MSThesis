# -*- coding: utf-8 -*-
"""
Created on Tue May 23 18:56:28 2023

@author: Edoardo Giancarli
"""

import numpy as np
import PyDeep_v1 as pdp

import torch
import torch.nn as nn
import torchvision 
from torchvision import transforms 
from torch.utils.data import Subset
from torch.utils.data import DataLoader


## Loading and preprocessing the data



## define model
model = pdp.GW_Deep()

## train model
dpt = pdp.DeepTools()
epochs = 20
lr = 1e-3

hist = dpt.train_model(epochs, lr, train_dataset, valid_dataset=valid_dataset)

## plot model loss and accuracy
dpt.show_model(hist[0], hist[1], mean_loss_valid=hist[2], mean_accuracy_valid=hist[3])

## test model


## save model



# end