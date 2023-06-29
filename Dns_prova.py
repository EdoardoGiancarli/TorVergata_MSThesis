# -*- coding: utf-8 -*-
"""
Created on Sat Jun  3 21:45:57 2023

@author: Edoardo Giancarli
"""

import PyDns_v1 as pds
from torchvision import transforms
import torch

# generate clean and noisy images
# %matplotlib inline

# pds.img_generator(N = 100,
#                   clean_imgs_path = "D:/Home/Universita'/Universita'/Magistrale/Master Thesis/Tesi_Codes/Prova_Denoiser/Model_Training/Clean_imgs/",
#                   noisy_imgs_path = "D:/Home/Universita'/Universita'/Magistrale/Master Thesis/Tesi_Codes/Prova_Denoiser/Model_Training/Noisy_imgs/")

# pds.img_generator(N = 100,
#                   clean_imgs_path = "D:/Home/Universita'/Universita'/Magistrale/Master Thesis/Tesi_Codes/Prova_Denoiser/Model_Test/Clean_imgs_test/",
#                   noisy_imgs_path = "D:/Home/Universita'/Universita'/Magistrale/Master Thesis/Tesi_Codes/Prova_Denoiser/Model_Test/Noisy_imgs_test/")


# define model
model = pds.DnsCNet()

# define dataset
# !!! remember to set the tensor dimension in the right way (NCHW) format: [batch_size, channels, image_heigth, image_width]
dnt = pds.DnsCNetTools()
clean_imgs = "D:/Home/Universita'/Universita'/Magistrale/Master Thesis/Tesi_Codes/Prova_Denoiser/Model_Training/Clean_imgs"
noisy_imgs = "D:/Home/Universita'/Universita'/Magistrale/Master Thesis/Tesi_Codes/Prova_Denoiser/Model_Training/Noisy_imgs"
batch_size = 7

# images to greyscale + torch tensor + resize
transform = transforms.Compose([transforms.functional.to_grayscale,
                                transforms.ToTensor(),
                                transforms.Resize((128, 128), antialias=True),])

# datasets
train_dataset, valid_dataset = dnt.make_dataset(noisy_imgs, clean_imgs, batch_size, transform=transform, valid_size=batch_size)

# train model
epochs = 2000
lr = 1e-3

torch.cuda.empty_cache()
train_loss, valid_loss = dnt.train_model(model, epochs, lr, train_dataset, valid_dataset)

# plot model loss and accuracy
dnt.show_model(comp_dloss = True)

# test model
clean_imgs_test = "D:/Home/Universita'/Universita'/Magistrale/Master Thesis/Tesi_Codes/Prova_Denoiser/Model_Test/Clean_imgs_test"
noisy_imgs_test = "D:/Home/Universita'/Universita'/Magistrale/Master Thesis/Tesi_Codes/Prova_Denoiser/Model_Test/Noisy_imgs_test"

test_dataset = dnt.make_dataset(noisy_imgs_test, clean_imgs_test, batch_size, transform=transform)
dnt.test_model(test_dataset)

# save model
filename = 'DnsCNet3.3_prova1_wNoiseImgs.pth'
dnt.save_model(batch_size, epochs, lr, filename)

# load model
filepath = "D:/Home/Universita'/Universita'/Magistrale/Master Thesis/Tesi_Codes/Prova_Denoiser/DNS_models/" + filename
model, obs = dnt.load_model(filepath)


# end