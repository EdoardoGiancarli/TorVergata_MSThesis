# -*- coding: utf-8 -*-
"""
Created on Fri Jul 21 14:39:33 2023

@author: Edoardo Giancarli
"""

import PyDRN as prn
from torchvision import transforms
import torch
# from tqdm import tqdm
import numpy as np
import matplotlib.pyplot as plt

# residual denoising network models
#
# models initialisation on a dataset subset

# warm up gpu
prn._warm_up_gpu().warm_up()

####################################   Residual denoising network models #########################################
#### define model
model = prn.DnsResNet(num_blocks=1, act_func='ReLU')

dnt = prn.DnsResNetTools()

#### define dataset
clean_imgs = "D:/Home/Universita'/Universita'/Magistrale/Master Thesis/Tesi_Codes/Prova_Denoiser/Model_Training/Clean_imgs"
noisy_imgs = "D:/Home/Universita'/Universita'/Magistrale/Master Thesis/Tesi_Codes/Prova_Denoiser/Model_Training/Noisy_imgs"
batch_size = 40

#### images to greyscale + torch tensor + resize
transform = transforms.Compose([#transforms.functional.to_grayscale,
                                transforms.ToTensor(),
                                transforms.Resize((150, 150), antialias=True),
                                prn.Normalize()])

#### datasets
train_dataset, valid_dataset = dnt.make_dataset(noisy_imgs, clean_imgs, batch_size, transform=transform, valid_size=7*batch_size)

#### train model and plot
# cascade learning
torch.cuda.empty_cache()
model, train_loss, valid_loss = dnt._quicker_train(1, 1, 0, model, 30, 1e-2, train_dataset, valid_dataset, stage_step=1,
                                                   act_func='ReLU', stoch_depth=False, prob = 0.5)

model, train_loss2, valid_loss2 = dnt._quicker_train(5, 1, 0, model, 30, 1e-3, train_dataset, valid_dataset, stage_step=1,
                                                     act_func='ReLU', stoch_depth=False, prob = 0.5)

# check model and total train and valid losses
print(model)
train = train_loss + train_loss2
valid = valid_loss + valid_loss2


#### plot model losses
x = np.arange(len(train))

fig = plt.figure(num = None, figsize = (12, 12), tight_layout = True)
ax = fig.add_subplot(111)
ax.plot(x, train, c = 'OrangeRed', label='train loss')
ax.scatter(x, valid, c = 'LawnGreen', label='valid. loss')
plt.ylabel('loss')
plt.title('Model mean loss through stages')         
plt.legend(loc = 'best')
ax.grid(True)
ax.label_outer()
ax.tick_params(which='both', direction='in',width=2)
ax.tick_params(which='major', direction='in',length=7)
ax.tick_params(which='minor', direction='in',length=4)
ax.xaxis.set_ticks_position('both')
ax.yaxis.set_ticks_position('both')
plt.show()


#### test model
clean_imgs_test = "D:/Home/Universita'/Universita'/Magistrale/Master Thesis/Tesi_Codes/Prova_Denoiser/Model_Test/Clean_imgs_test"
noisy_imgs_test = "D:/Home/Universita'/Universita'/Magistrale/Master Thesis/Tesi_Codes/Prova_Denoiser/Model_Test/Noisy_imgs_test"

test_dataset = dnt.make_dataset(noisy_imgs_test, clean_imgs_test, batch_size, transform=transform)
dnt.test_model(test_dataset)


#### save model
filename = 'DnsResNet_model12.pth'
notes = "Base6CNN with ReLU, bias, Gauss filters, [0, 1], no Stochastic Depth. Training takes about 27min."
gpu = '5003MiB / 6144MiB'

dnt.save_model(batch_size=40, epochs=[30, 30, 50, 50], learning_rate=[1e-2, 1e-3, 1e-4, 1e-5], stages=[0, 7, 888888, 888], activation='ReLU',
               filename=filename, train_loss=train, valid_loss=valid, notes=notes, gpu=gpu, start_resblocks=[1, 'ReLU'])

#### load model
model, obs = dnt.load_model('DnsResNet_model2.pth')



#################################### models initialisation on a dataset subset #########################################
import PyDRN as prn
from torchvision import transforms
import torch

#### define subsets
# !!! remember to set the tensor dimension in the right way (NCHW) format: [batch_size, channels, image_heigth, image_width]
dnt = prn.DnsResNetTools()
clean_imgs = "D:/Home/Universita'/Universita'/Magistrale/Master Thesis/Tesi_Codes/Prova_Denoiser/Model_Training/Init_clean_subset"
noisy_imgs = "D:/Home/Universita'/Universita'/Magistrale/Master Thesis/Tesi_Codes/Prova_Denoiser/Model_Training/Init_noisy_subset"
batch_size = 5

#### images to greyscale + torch tensor + resize
transform = transforms.Compose([#transforms.functional.to_grayscale,
                                transforms.ToTensor(),
                                transforms.Resize((128, 128), antialias=True),
                                prn.Normalize()])

#### subsets
valid_dataset, train_subset = dnt.make_dataset(noisy_imgs, clean_imgs, batch_size, transform=transform, valid_size=2*batch_size)

#### initialize model
model = prn.DnsResNet(num_blocks=1, bias=True, act_func='PReLU')

epochs = 20
lr = 1e-2

# train
torch.cuda.empty_cache()

model, train_loss, valid_loss = dnt._quicker_train(1, 1, 0, model, 40, 1e-2, train_subset, valid_dataset,
                                                   bias=True, act_func='PReLU', plot=False)

# plot model loss and accuracy
dnt.show_model(comp_dloss = True)
