# -*- coding: utf-8 -*-
"""
Created on Fri Jun 30 22:13:47 2023

@author: Edoardo Giancarli
"""

import torch         
from torchvision import transforms
import PyDeep as pdp
import PyDRN as prn

# residual classifier network model


# warm up gpu
# pdp._warm_up_gpu().warm_up()

#### define model
denoiser_model = 'DnsResNet_model16sistis.pth'
model = pdp.GWResNet(num_blocks=0, act_func='PReLU')

# call DeepTools
dpt = pdp.GWDeepTools()

# dpt._check_gpumemory_usage(model, epochs=3, lr=1e-4, batch_size=80, stage=0)

#### define images transform
transform = transforms.Compose([transforms.ToTensor(),
                                transforms.Resize((150, 150), antialias=True),
                                pdp.Normalize(),
                                pdp.Denoiser(denoiser_model, renormalisation=None),
                                pdp.RGB2Grayscale(renormalisation='unilateral')])



#### define dataset
# case_study = 'MilkyWay'
# case_study = 'SubGroup'
# case_study = 'LocalGroup'
# case_study = 'Mpc_scale'
case_study = 'macro_test'


batch_size = 100
train_path, val, test_path = pdp._choose_dataset(case_study, batch_size)

# datasets
# train_dataset, valid_dataset = dpt.make_dataset(train_path, batch_size, transform=transform,
#                                                 valid_size=val*batch_size)

try:
    train_dataset = dpt.manage_dataset(dataset_name='train_dataset_'+case_study+'_model21_'+'3'+'.pt', mode='load')
    valid_dataset = dpt.manage_dataset(dataset_name='valid_dataset_'+case_study+'_model21_'+'3'+'.pt', mode='load')
except:
    train_dataset, valid_dataset = dpt.make_dataset(train_path, batch_size, transform=transform,
                                                    valid_size=val*batch_size)


#### train model
# cascade learning
# res_act = 'PReLU'

# epochs = [5, 5, 5]
# lr = [1e-3, 5e-4, 1e-4]

# epochs = [6, 6, 6]
# lr = [1e-3, 5e-4, 1e-4]

# epochs = [6, 6, 6]
# lr = [1e-3, 5e-4, 1e-4]

# epochs = [6, 6, 6]
# lr = [1e-3, 5e-4, 1e-4]

epochs = [7, 7, 7]
lr = [1e-3, 5e-4, 1e-4]



torch.cuda.empty_cache()

model, tl, ta, vl, va = dpt._quicker_train(2, 1, 0, model, epochs[0], lr[0], train_dataset, valid_dataset, stage_step=0)

model, tl2, ta2, vl2, va2 = dpt._quicker_train(2, 1, 0, model, epochs[1], lr[1], train_dataset, valid_dataset, stage_step=0)

model, tl3, ta3, vl3, va3 = dpt._quicker_train(2, 1, 0, model, epochs[2], lr[2], train_dataset, valid_dataset, stage_step=0)


#### plot model loss and accuracy
# print(model)
train_loss = tl + tl2 + tl3 #+ tl4 #+ tl5 #+ tl6
train_accuracy =  ta + ta2 + ta3 #+ ta4 #+ ta5 #+ ta6

valid_loss = vl + vl2 + vl3 #+ vl4 #+ vl5 #+ vl6
valid_accuracy = va + va2 + va3 #+ va4 #+ va5 #+ va6

# loss and accuracy plot
dpt.show_model(comp_dloss=False, stage=0, train_loss=train_loss, train_accuracy=train_accuracy,
               valid_loss=valid_loss, valid_accuracy=valid_accuracy, title_notes='- Case study: ' + case_study)


#### test model
# test_dataset = dpt.make_dataset(test_path, batch_size, transform=transform)

try:
    test_dataset = dpt.manage_dataset(dataset_name='test_dataset_'+case_study+'_model21_'+'3'+'.pt', mode='load')
except:
    test_dataset = dpt.make_dataset(test_path, batch_size, transform=transform)

dpt.test_model(test_dataset)


#### save model
filename = 'GWResNet_model' + '21' + '_finaltest' + '3' + '_' + case_study + '.pth'
notes = "eventual notes"
gpu = "gpu memory usage"

dpt.save_model(batch_size=batch_size, epochs=epochs, learning_rate=lr, stages=['00', '00', '00'], activation=None,
               filename=filename, denoiser=denoiser_model, train_loss=train_loss, train_accuracy=train_accuracy,
               valid_loss=valid_loss, valid_accuracy=valid_accuracy, notes=notes, gpu=gpu, start_resblocks=None)


# save datasets
dpt.manage_dataset(train_dataset, 'train_dataset_'+case_study+'_model21_'+'3', mode='save')
dpt.manage_dataset(valid_dataset, 'valid_dataset_'+case_study+'_model21_'+'3', mode='save')
dpt.manage_dataset(test_dataset, 'test_dataset_'+case_study+'_model21_'+'3', mode='save')


#### load model
dpt = pdp.GWDeepTools()

case_study = 'MilkyWay'

cls_model, obs = dpt.load_model('GWResNet_model' + '21' + '_finaltest' + '3' + '_' + case_study + '.pth')
dns_model, dns_obs = prn.DnsResNetTools().load_model('DnsResNet_model18_0.pth')



# end