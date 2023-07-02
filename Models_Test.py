# -*- coding: utf-8 -*-
"""
Created on Sun Jun 11 18:20:16 2023

@author: Edoardo Giancarli
"""

import PyDeep as pdp
import PyDns as pds
import numpy as np
import torch
from torchvision import transforms
import torch.nn as nn
from PIL import Image
import matplotlib.pyplot as plt
import cv2

#### denoiser test ###############################################################
# import clean, noisy and noise images
clean_img = Image.open("D:/Home/Universita'/Universita'/Magistrale/Master Thesis/Tesi_Codes/Prova_Denoiser/Model_Test/Clean_imgs_test/clean_spectr0.png")
noisy_img = Image.open("D:/Home/Universita'/Universita'/Magistrale/Master Thesis/Tesi_Codes/Prova_Denoiser/Model_Test/Noisy_imgs_test/noisy_spectr0.png")
noise_img = Image.open("D:/Home/Universita'/Universita'/Magistrale/Master Thesis/Tesi_Codes/Prova_Denoiser/Model_Test/Clean_imgs_test/noise_spectrogram_187.png")

# display images
nxd = clean_img.size[0]
pds.cv2disp(img_name='Clean', img=cv2.resize(np.array(clean_img), (3*nxd, 3*nxd)))
pds.cv2disp(img_name='Noisy', img=cv2.resize(np.array(noisy_img), (3*nxd, 3*nxd)), x_pos = int(3*nxd))
pds.cv2disp(img_name='Noise', img=cv2.resize(np.array(noise_img), (3*nxd, 3*nxd)), x_pos = int(6*nxd))

# load model (from DNS_models directory)
dnt = pds.DnsCNetTools()

filename = 'DnsCNet3_prova2_wNoiseImgs.pth'
filepath = "D:/Home/Universita'/Universita'/Magistrale/Master Thesis/Tesi_Codes/Prova_Denoiser/DNS_models/" + filename
model, obs = dnt.load_model(filepath)

# images to greyscale + torch tensor + resize
transform = transforms.Compose([transforms.functional.to_grayscale,
                                transforms.ToTensor(),
                                transforms.Resize((128, 128), antialias=True),])

# model outputs
noisy_out = model(transform(noisy_img).unsqueeze(0)).detach().numpy().squeeze()
noise_out = model(transform(noise_img).unsqueeze(0)).detach().numpy().squeeze()

pds.cv2disp(img_name='Model output: noisy', img=cv2.resize(noisy_out, (3*nxd, 3*nxd)), x_pos = int(9*nxd))
pds.cv2disp(img_name='Model output: noise', img=cv2.resize(noise_out, (3*nxd, 3*nxd)), x_pos = 12*nxd)

# test on the differences between true and pred
true_noisy = transform(noisy_img).detach().numpy().squeeze()
true_noise = transform(noise_img).detach().numpy().squeeze()

loss_noisy = np.median(noisy_out - true_noisy) # O(1e-3)
loss_noise = np.median(noise_out - true_noise) # O(1e-3)


#### cnn test ####################################################################







# end