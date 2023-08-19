# -*- coding: utf-8 -*-
"""
Created on Fri Jul 21 14:39:33 2023

@author: Edoardo Giancarli
"""

import PyDRN as prn
from torchvision import transforms
import torch
import numpy as np
import matplotlib.pyplot as plt

# residual denoising network model
#
# models initialisation on a dataset subset


# warm up gpu
# prn._warm_up_gpu().warm_up()

####################################   Residual denoising network models #########################################
#### define model
model = prn.DnsResNet(num_blocks=1, act_func='ReLU')

dnt = prn.DnsResNetTools()

#### define dataset
# !!! remember to set the tensor dimension in the right way (NCHW) format: [batch_size, channels, image_heigth, image_width]
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

model, train_loss2, valid_loss2 = dnt._quicker_train(2, 2, 1, model, 30, 1e-3, train_dataset, valid_dataset, stage_step=1,
                                                     act_func='ReLU', stoch_depth=False, prob = 0.5)

model, train_loss3, valid_loss3 = dnt._quicker_train(2, 2, 5, model, 30, 1e-3, train_dataset, valid_dataset, stage_step=1,
                                                     act_func='ReLU', stoch_depth=False, prob = 0.5)

model, train_loss4, valid_loss4 = dnt._quicker_train(2, 2, 9, model, 30, 1e-4, train_dataset, valid_dataset, stage_step=0,
                                                     act_func='ReLU', stoch_depth=False, prob = 0.5)

# model, train_loss5, valid_loss5 = dnt._quicker_train(2, 2, 8, model, 30, 1e-4, train_dataset, valid_dataset, stage_step=0,
#                                                      act_func='ReLU', stoch_depth=False, prob = 0.5)

# model, train_loss6, valid_loss6 = dnt._quicker_train(3, 1, 8, model, 50, 1e-5, train_dataset, valid_dataset, stage_step=0,
#                                                       act_func='ReLU', stoch_depth=False, prob = 0.5)



# epochs = 100
# learn_rate = 1e-4
# model, train, valid = dnt.train_model(model, epochs, learn_rate, train_dataset, valid_dataset, stage = None,
#                                       bias = True, act_func = 'PReLU', stoch_depth = True, prob = 0.5)


print(model)
train = train_loss + train_loss2 + train_loss3 + train_loss4 #+ train_loss5 #+ train_loss6
valid = valid_loss + valid_loss2 + valid_loss3 + valid_loss4 #+ valid_loss5 #+ valid_loss6

# loss plot
dnt.show_model(comp_dloss = True, stage=9, train_loss=train, valid_loss=valid)

# test model
clean_imgs_test = "D:/Home/Universita'/Universita'/Magistrale/Master Thesis/Tesi_Codes/Prova_Denoiser/Model_Test/Clean_imgs_test"
noisy_imgs_test = "D:/Home/Universita'/Universita'/Magistrale/Master Thesis/Tesi_Codes/Prova_Denoiser/Model_Test/Noisy_imgs_test"

test_dataset = dnt.make_dataset(noisy_imgs_test, clean_imgs_test, batch_size, transform=transform)
dnt.test_model(test_dataset)


#### save model
filename = 'DnsResNet_model13.pth'
notes = "Base6CNN with ReLU, bias, Gauss filters, [0, 1], no Stochastic Depth. Stage 9 trained more."
gpu = '4706MiB / 6144MiB'

dnt.save_model(batch_size=40, epochs=30, learning_rate=[1e-2, 1e-3, 1e-3, 1e-4], stages=[0, 5, 4, 9999], activation='ReLU',
               filename=filename, train_loss=train, valid_loss=valid, notes=notes, gpu=gpu, start_resblocks=[1, 'ReLU'])

#### load model
model, obs = dnt.load_model('DnsResNet_model13.pth')




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
transform = transforms.Compose([transforms.functional.to_grayscale,
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



############################################
import PyDRN as prn
from torchvision import transforms
import torch
import torch.nn as nn

import numpy as np
import matplotlib.pyplot as plt

dnt = prn.DnsResNetTools()


models = []
obs = []

filenames = ['DnsResNet_Prova' + str(i + 1) + '.pth' for i in range(5)]

for m in filenames:
    mod, ob = dnt.load_model(m)
    models.append(mod)
    obs.append(ob)

for i, name in zip(models, filenames):
    print(f'#######################  model {name}  ###########################')
    print(i)
    print('##########################################################')



for module in models[2].modules():
    print(module)




current_block_index = 0
target_block_index = 10
residual_block_params = None
# mod = prn.ResBlock
mod = nn.Sequential

for module in models[2].modules():
    
    # Check if the module is an instance of the residual block you defined in your ResNet class.
    if isinstance(module, mod):
        
        # Check if the current block index matches the target block index
        if current_block_index == target_block_index:
            
            # Access the residual block here (stored in the 'module' variable)
            print(f"{mod} module {target_block_index}:")
            print(module)
        
            # Access the parameters of the second residual block here
            residual_block_params = list(module.parameters())
            break
        
        # Increment the current block index
        current_block_index += 1


model_parameters_numpy = [param.detach().numpy() for param in residual_block_params]

squeezed_params = [p[0, 0, :, :] if (len(p.shape) == 4) else p for p in model_parameters_numpy]



for i in squeezed_params:
    if len(i.shape) == 2:
        
        plt.figure(None, tight_layout=True)
        a = plt.imshow(i, cmap='Greys', vmin=-1, vmax=1)
        plt.colorbar(a)
        plt.show()



a = [b for b in train_dataset]

dataset_numpy = [[input_data.numpy(), target_data.numpy()] for input_data, target_data in train_dataset]

c0_in0 = dataset_numpy[4][0][0, 0, :, :]
c0_out0 = dataset_numpy[4][1][0, 0, :, :]

c0_in5 = dataset_numpy[5][0][5, 0, :, :]
c0_out5 = dataset_numpy[5][1][5, 0, :, :]

c3_in0 = dataset_numpy[6][0][0, 0, :, :]
c3_out0 = dataset_numpy[6][1][0, 0, :, :]


for a in [c0_in0, c0_out0, c0_in5, c0_out5, c3_in0, c3_out0]:
    
    plt.figure(None)
    o = plt.imshow(a, vmin=-1, vmax=1)
    plt.colorbar(o)
    plt.show()


