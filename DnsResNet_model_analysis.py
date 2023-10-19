# -*- coding: utf-8 -*-
"""
Created on Fri Jul 21 14:39:33 2023

@author: Edoardo Giancarli
"""

import PyDRN as prn
from torchvision import transforms
import torch

# residual denoising network model


# warm up gpu
# prn._warm_up_gpu().warm_up()

####################################   Residual denoising network models #########################################
#### define model
model = prn.DnsResNet(num_blocks=1, act_func='ReLU')

dnt = prn.DnsResNetTools()

# dnt._check_gpumemory_usage(model, epochs=3, lr=1e-4, batch_size=50, stage=0)

#### define dataset
# !!! remember to set the tensor dimension in the right way (NCHW) format: [batch_size, channels, image_heigth, image_width]
clean_imgs = "clean_imgs_path"
noisy_imgs = "noisy_imgs_path"
batch_size = 40

#### images to greyscale + torch tensor + resize
transform = transforms.Compose([transforms.ToTensor(),
                                transforms.Resize((150, 150), antialias=True),
                                prn.Normalize()])


#### datasets
train_dataset, valid_dataset = dnt.make_dataset(noisy_imgs, clean_imgs, batch_size, transform=transform, valid_size=12*batch_size)


#### train model and plot
# cascade learning
torch.cuda.empty_cache()

epochs = 30
lr = [0.01, 0.001, 0.001, 0.0001]

model, train_loss, valid_loss = dnt._quicker_train(1, 1, 0, model, epochs, lr[0], train_dataset, valid_dataset, stage_step=1,
                                                   act_func='ReLU', stoch_depth=False, prob=0)

model, train_loss2, valid_loss2 = dnt._quicker_train(2, 2, 1, model, epochs, lr[1], train_dataset, valid_dataset, stage_step=1,
                                                     act_func='ReLU', stoch_depth=False, prob=0)

model, train_loss3, valid_loss3 = dnt._quicker_train(2, 2, 5, model, epochs, lr[2], train_dataset, valid_dataset, stage_step=1,
                                                     act_func='ReLU', stoch_depth=False, prob=0)


# save checkpoint!!!
try:
    train = train_loss+train_loss2+train_loss3
    valid = valid_loss+valid_loss2+valid_loss3
except:
    train = train_loss+train_loss2
    valid = valid_loss+valid_loss2

dnt.save_model(batch_size='temp', epochs='temp', learning_rate='temp', stages='temp', activation='temp',
               filename='ztemp_model_checkpoint.pth', train_loss=train, valid_loss=valid)

model, obs = dnt.load_model('ztemp_model_checkpoint.pth', set_classmodel=True)
train = obs['tot_loss_train']
valid = obs['tot_loss_valid'] 



model, train_loss4, valid_loss4 = dnt._quicker_train(2, 2, 8, model, epochs, lr[3], train_dataset, valid_dataset, stage_step=0,
                                                     act_func='ReLU', stoch_depth=False, prob=0)


print(model)
train = train_loss + train_loss2 + train_loss3 + train_loss4 #+ train_loss5 #+ train_loss6
valid = valid_loss + valid_loss2 + valid_loss3 + valid_loss4 #+ valid_loss5 #+ valid_loss6

# loss plot
dnt.show_model(comp_dloss = False, stage=8, train_loss=train, valid_loss=valid)

# test model
clean_imgs_test = "D:/Home/Universita'/Universita'/Magistrale/Master Thesis/Tesi_Codes/Prova_Denoiser/Model_Test/Clean_imgs_test"
noisy_imgs_test = "D:/Home/Universita'/Universita'/Magistrale/Master Thesis/Tesi_Codes/Prova_Denoiser/Model_Test/Noisy_imgs_test"

test_dataset = dnt.make_dataset(noisy_imgs_test, clean_imgs_test, batch_size, transform=transform)
dnt.test_model(test_dataset)


#### save model
filename = 'DnsResNet_model18quater.pth'
notes = "eventual notes"
gpu = "gpu usage"

dnt.save_model(batch_size=batch_size, epochs=epochs, learning_rate=lr, stages=[0, 4, 4, 8888],
               activation='ReLU', filename=filename, train_loss=train, valid_loss=valid, notes=notes, gpu=gpu, start_resblocks=[1, 'ReLU'])

#### load model
dnt = prn.DnsResNetTools()
model, obs = dnt.load_model('DnsResNet_model16sistis.pth', set_classmodel=False)
train = obs['tot_loss_train']
valid = obs['tot_loss_valid']


# end