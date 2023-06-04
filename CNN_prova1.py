# -*- coding: utf-8 -*-
"""
Created on Tue May 23 18:56:28 2023

@author: Edoardo Giancarli
"""

# import numpy as np
# import matplotlib.pyplot as plt

import torch
# from torch.autograd import profiler
from torchvision import transforms 

import PyDeep_v1 as pdp
# import time


#### define model
model = pdp.GW_Deep()


#### define dataset
## !!! remember to set the tensor dimension in the right way (NCHW) format: [batch_size, channels, image_heigth, image_width]
dpt = pdp.DeepTools()
data_path = "D:/Home/Universita'/Universita'/Magistrale/Master Thesis/Tesi_Codes/Prova_cnn/Train_Images"
img_height, img_width = 128, 128
batch_size = 10

# define images transform
transform = transforms.Compose([transforms.functional.to_grayscale,
                                transforms.ToTensor(),
                                transforms.Resize((img_height, img_width), antialias=True),])

# datasets
train_dataset, valid_dataset = dpt.make_dataset(data_path, batch_size, transform=transform, valid_size=4*batch_size)


#### train model
epochs = 40
lr = 5e-5

torch.cuda.empty_cache()
hist = dpt.train_model(model, epochs, lr, train_dataset, valid_dataset=valid_dataset)

# 2min 40s with cpu, 20 epochs, 16 img for train, 4 for validation, batch_size of 4
# 26s with gpu, 20 epochs, 16 img for train, 2 for validation, batch_size of 2

import numpy as np
for i in range(4):
    print(hist[i][-1])

#### plot model loss and accuracy
dpt.show_model()


#### test model
dpt.test_model()


#### save model
filename = 'CNNmodel_prova1.pth'
dpt.save_model(batch_size, epochs, filename)


#### load model
filepath = "D:/Home/Universita'/Universita'/Magistrale/Master Thesis/Tesi_Codes/Prova_cnn/" + filename
model, obs = dpt.load_model(filepath)


# end