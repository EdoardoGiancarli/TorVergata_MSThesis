# -*- coding: utf-8 -*-
"""
Created on Thu Jun  1 23:27:35 2023

@author: Edoardo Giancarli
"""

import numpy as np
import PyDns_v1 as pds
# import PyDeep_v1 as pdp
import torch
from torchvision import transforms
import torch.nn as nn
from PIL import Image
import matplotlib.pyplot as plt
import cv2
from tqdm import tqdm

############  1st part  ############
# simple cnn
class Conv_NxN(nn.Module):
    
    def __init__(self, kernel_size):
        
        super(Conv_NxN, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=1, kernel_size=kernel_size, padding=int(kernel_size/2), bias=False)
        #self.conv1.weight.data.fill_(1./kernel_size**2)
    
    def forward(self, x):
        
        x = self.conv1(x)
        
        return x


# generate clean and noisy images
# !!! %matplotlib inline
pds.img_generator(100)

# import clean and noisy images
clean_img = Image.open("D:/Home/Universita'/Universita'/Magistrale/Master Thesis/Tesi_Codes/Prova_Denoiser/Clean_imgs/clean_spectr0.png")
noisy_img = Image.open("D:/Home/Universita'/Universita'/Magistrale/Master Thesis/Tesi_Codes/Prova_Denoiser/Noisy_imgs/noisy_spectr0.png")

# display images
nxd = clean_img.size[0]
pds.cv2disp(img_name='Clean', img=clean_img)
pds.cv2disp(img_name='Noisy', img=noisy_img, x_pos = 1.5*nxd)

# images to greyscale + torch tensor + resize
transform = transforms.Compose([transforms.functional.to_grayscale,
                                transforms.ToTensor(),
                                transforms.Resize((128, 128), antialias=True),])

clean_img_tch = transform(clean_img)
noisy_img_tch = transform(noisy_img)

# apply conv
conv_kernel = Conv_NxN(31)
output_cnn_tch = conv_kernel(noisy_img_tch)

# display output and kernel
output_cnn_np = output_cnn_tch.detach().numpy().squeeze()
pds.cv2disp(img_name='Output convolution', img=output_cnn_np, x_pos = 3*nxd)

kernel_tch = list(conv_kernel.parameters())[0]
kernel_np = kernel_tch.detach().numpy().squeeze()
pds.cv2disp(img_name='Convolution kernel', img=cv2.resize(kernel_np, (nxd, nxd)), x_pos = 3*nxd)




############  2nd part  ############
import numpy as np
import PyDns_v1 as pds
# import PyDeep_v1 as pdp
import torch
from torchvision import transforms
import torch.nn as nn
from PIL import Image
import matplotlib.pyplot as plt
import cv2
from tqdm import tqdm


# simple cnn
class CdsPy_1(nn.Module):
    
    def __init__(self):
        super(CdsPy_1, self).__init__()
        
        self.layers = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=8, kernel_size=7, padding=3), nn.PReLU(),
            nn.Conv2d(in_channels=8, out_channels=8, kernel_size=7, padding=3), nn.PReLU(),
            nn.Conv2d(in_channels=8, out_channels=8, kernel_size=7, padding=3), nn.PReLU(),
            nn.Conv2d(in_channels=8, out_channels=8, kernel_size=7, padding=3), nn.PReLU(),
            nn.Conv2d(in_channels=8, out_channels=1, kernel_size=7, padding=3), nn.PReLU())
    
    
    def forward(self, x):
        
        x = self.layers(x)
        
        return x


# train denoiser function
def train(epochs, input_img, target, comp_dloss = False):
    
    model = CdsPy_1()
    
    if torch.cuda.is_available():
        device = torch.device("cuda:0")
        print("Training on GPU...")
        model = model.to(device)
        input_img = input_img.to(device)
        target = target.to(device)
    
    else:
        print("No GPU available, redirecting to CPU...\n")
        user_input = input("Continue training on CPU? (y/n): ")
        
        if user_input.lower() == "n":
            raise Exception("Training interrupted")
        else:
            device = torch.device("cpu")
    
    loss_fn = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr = 3e-3)
    
    loss_train = []
    store_every = 25
    nxd = 512
        
    for epoch in tqdm(range(epochs)):
        pred = model(input_img)
        
        optimizer.zero_grad()
        
        loss = loss_fn(pred, target)
        loss.backward()
        optimizer.step()
                
        pred_np = pred.detach().cpu().numpy().squeeze()
        cv2.putText(pred_np, f'Ep {epoch + 1}', (2, 30), 0, 0.2, int(np.max(pred_np) + 1), 1, cv2.LINE_AA)
        pds.cv2disp(img_name='Output convolution', img=cv2.resize(pred_np, (nxd, nxd)), x_pos = 2*nxd)
        cv2.waitKey(1)        
        
        if epoch % store_every == 0:
            loss_train.append(loss.item())
    
    plt.figure(num = None)
    plt.plot(store_every*np.arange(len(loss_train)), loss_train, c = 'OrangeRed')
    plt.title('Loss')
    plt.ylabel('loss values')
    plt.xlabel('epoch')
    plt.show()
    
    if comp_dloss:
        dloss = [loss[l+1]-loss[l] for l in range(len(loss)-1)]

        plt.figure(num = None)
        plt.plot(store_every*np.arange(len(dloss)), dloss, c = 'OrangeRed')
        plt.title('Loss change rate')
        plt.ylabel('loss change rate values')
        plt.xlabel('epoch')
        plt.show()
    
    return loss_train

# import clean and noisy images
clean_img = Image.open("D:/Home/Universita'/Universita'/Magistrale/Master Thesis/Tesi_Codes/Prova_Denoiser/Clean_imgs/clean_spectr0.png")#.convert('L')
noisy_img = Image.open("D:/Home/Universita'/Universita'/Magistrale/Master Thesis/Tesi_Codes/Prova_Denoiser/Noisy_imgs/noisy_spectr0.png")#.convert('L')

# images to greyscale + torch tensor + resize
transform = transforms.Compose([transforms.functional.to_grayscale,
                                transforms.ToTensor(),
                                transforms.Resize((128, 128), antialias=True),])

# images in torch
clean_img_tch = transform(clean_img)
noisy_img_tch = transform(noisy_img)

# display images
nxd = 512
pds.cv2disp(img_name='Clean', img=cv2.resize(clean_img_tch.detach().numpy().squeeze(), (nxd, nxd)), x_pos = 0)
pds.cv2disp(img_name='Noisy', img=cv2.resize(noisy_img_tch.detach().numpy().squeeze(), (nxd, nxd)), x_pos = 1*nxd)

# train
loss = train(2000, noisy_img_tch, clean_img_tch)
dloss = [loss[l+1]-loss[l] for l in range(len(loss)-1)]

plt.figure(num = None)
plt.plot(25*np.arange(len(dloss)), dloss, c = 'OrangeRed')
plt.title('Loss change rate')
plt.ylabel('loss change rate values')
plt.xlabel('epoch')
plt.show()



####################################  3rd part  ########################################################################
import PyDns_v1 as pds
from torchvision import transforms
import torch

# generate clean and noisy images
# !!! %matplotlib inline
pds.img_generator(100)

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
clean_imgs_test = ""
noisy_imgs_test = ""

test_dataset = dnt.make_dataset(noisy_imgs_test, clean_imgs_test, batch_size, transform=transform)
dnt.test_model()

# save model
filename = 'DnsCNet5_prova1.pth'
dnt.save_model(batch_size, epochs, lr, filename)

# load model
filepath = "D:/Home/Universita'/Universita'/Magistrale/Master Thesis/Tesi_Codes/Prova_Denoiser/" + filename
model, obs = dnt.load_model(filepath)

