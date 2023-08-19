# -*- coding: utf-8 -*-
"""
Created on Sun Jul 23 16:03:39 2023

@author: Edoardo Giancarli
"""

import PyDRN as prn
import numpy as np
import PyUtils as pu
# import torch
from torchvision import transforms
# import torch.nn as nn
from PIL import Image
# import matplotlib.pyplot as plt
import cv2

# import clean, noisy and noise images
clean_imgs, noisy_imgs, noise_clean_imgs, noise_noisy_imgs = [], [], [], []
ind1, ind2 = [], []

# spectrograms
for _ in range(1000):
    index1 = np.random.randint(0, 1210)
    
    if len(clean_imgs) == 10:
        break
    else:
        try:
            clean_imgs.append(Image.open("D:/Home/Universita'/Universita'/Magistrale/Master Thesis/Tesi_Codes/Prova_Denoiser/Model_Test/Clean_imgs_test/clean_spectr_" + str(index1) + ".png").convert("RGB"))
            noisy_imgs.append(Image.open("D:/Home/Universita'/Universita'/Magistrale/Master Thesis/Tesi_Codes/Prova_Denoiser/Model_Test/Noisy_imgs_test/noisy_spectr_" + str(index1) + ".png").convert("RGB"))
            ind1.append(index1)
        except:
            pass

# noise
for _ in range(1000):
    index2 = np.random.randint(0, 286)
    
    if len(noise_clean_imgs) == 10:
        break
    else:
        try:
            noise_clean_imgs.append(Image.open("D:/Home/Universita'/Universita'/Magistrale/Master Thesis/Tesi_Codes/Prova_Denoiser/Model_Test/Clean_imgs_test/znoise_cleanspectr_" + str(index2) + ".png").convert("RGB"))
            noise_noisy_imgs.append(Image.open("D:/Home/Universita'/Universita'/Magistrale/Master Thesis/Tesi_Codes/Prova_Denoiser/Model_Test/Noisy_imgs_test/znoise_noisyspectr_" + str(index2) + ".png").convert("RGB"))
            ind2.append(index2)
        except:
            pass


# display images
nxd = clean_imgs[0].size[0]
for i, a in enumerate(ind1):
    pu.cv2disp(img_name='Clean spectr number ' + str(a), img=cv2.resize(np.array(clean_imgs[i]), (3*nxd, 3*nxd)))
    pu.cv2disp(img_name='Noisy spectr number ' + str(a), img=cv2.resize(np.array(noisy_imgs[i]), (3*nxd, 3*nxd)), x_pos = 3*nxd)

for j, b in enumerate(ind2):
    pu.cv2disp(img_name='Noise cleanspectr number ' + str(b), img=cv2.resize(np.array(noise_clean_imgs[j]), (3*nxd, 3*nxd)), x_pos = 6*nxd)
    pu.cv2disp(img_name='Noise noisyspectr number ' + str(b), img=cv2.resize(np.array(noise_noisy_imgs[j]), (3*nxd, 3*nxd)), x_pos = 9*nxd)


# images to torch tensor + resize + normalisation
transform = transforms.Compose([#transforms.functional.to_grayscale,
                                transforms.ToTensor(),
                                transforms.Resize((150, 150), antialias=True),
                                prn.Normalize()])

# transformed imgs
torch_clean_imgs = [transform(clean).detach().numpy().squeeze() for clean in clean_imgs]
torch_noisy_imgs = [transform(noisy).detach().numpy().squeeze() for noisy in noisy_imgs]
torch_noiseclean_imgs = [transform(cnoise).detach().numpy().squeeze() for cnoise in noise_clean_imgs]
torch_noisenoisy_imgs = [transform(nnoise).detach().numpy().squeeze() for nnoise in noise_noisy_imgs]



############################################ ResNet denoisers performance test ############################################

# load model (from DNS_models directory)
dnt_resnet = prn.DnsResNetTools()

filename = 'DnsResNet_model13.pth'
model, obs = dnt_resnet.load_model(filename)

# model outputs
noisy_out = [model(transform(noisy_img).unsqueeze(0)).detach().numpy().squeeze() for noisy_img in noisy_imgs]
noise_out = [model(transform(znoise_img).unsqueeze(0)).detach().numpy().squeeze() for znoise_img in noise_noisy_imgs]

for i, ind in enumerate(ind1):
    # take only 2d array for display
    noisy = np.log10(np.sum(noisy_out[i], axis=0) + 1e-12)
    pu.cv2disp(img_name='Model output: noisy ' + str(ind), img=cv2.resize(noisy, (3*nxd, 3*nxd)), x_pos = 6*nxd)

for i, ind in enumerate(ind2):
    # take only 2d array for display
    noise = np.log10(np.sum(noise_out[i], axis=0) + 1e-12)
    pu.cv2disp(img_name='Model output: noise ' + str(ind), img=cv2.resize(noise, (3*nxd, 3*nxd)), x_pos = 9*nxd)

# test on the performance

diff_clean_outmodel = [np.abs(np.median(clean - noisy))
                       for clean, noisy in zip(torch_clean_imgs, noisy_out)]         # difference between clean and model output

diff_cnoise_outmodel = [np.abs(np.median(clean - noisy))
                       for clean, noisy in zip(torch_noiseclean_imgs, noise_out)]    # difference between clean noise and model output



# end