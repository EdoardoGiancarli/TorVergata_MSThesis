# -*- coding: utf-8 -*-
"""
Created on Tue May 23 18:56:28 2023

@author: Edoardo Giancarli
"""

# import numpy as np
# import matplotlib.pyplot as plt
import torch
from torchvision import transforms
import PyDeep_v2 as pdp
# from torch.autograd import profiler


#### define model
model = pdp.GW_Deep(denoiser = True, denoiser_model = 'DnsCNet3_prova2_wNoiseImgs.pth')


#### define dataset
# remember to set the tensor dimension in the right way (NCHW) format: [batch_size, channels, image_heigth, image_width]
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
epochs = 200
lr = 1e-4

torch.cuda.empty_cache()
hist = dpt.train_model(model, epochs, lr, train_dataset, valid_dataset)

# 2min 40s with cpu, 20 epochs, 16 img for train, 4 for validation, batch_size of 4
# 26s with gpu, 20 epochs, 16 img for train, 2 for validation, batch_size of 2


#### plot model loss and accuracy
dpt.show_model()


#### test model
test_imgs = "D:/Home/Universita'/Universita'/Magistrale/Master Thesis/Tesi_Codes/Prova_cnn/Test_Images/"

test_dataset = dpt.make_dataset(test_imgs, batch_size, transform=transform)
dpt.test_model(test_dataset)


#### save model
filename = 'CNNmodel7_1_prova1_wDnsCNet3.pth'
dpt.save_model(batch_size, epochs, lr, filename, denoiser = 'DnsCNet3_prova2_wNoiseImgs.pth')


#### load model
filepath = "D:/Home/Universita'/Universita'/Magistrale/Master Thesis/Tesi_Codes/Prova_cnn/CNN_models/" + filename
model, obs = dpt.load_model(filepath)


# end