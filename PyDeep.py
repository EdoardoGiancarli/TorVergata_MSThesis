# -*- coding: utf-8 -*-
"""
Created on Sat Aug 19 00:14:10 2023

@author: Edoardo Giancarli
"""

#### ModuleMSThesis - CNN for spectrograms classification version 5 ###############################################

####   libraries   #####

import numpy as np                              # operations
# import pandas as pd                             # dataframe 
import random
import pathlib                                  # filepaths
import os                    
from PIL import Image                           # images
from tqdm import tqdm                           # loop progress bar

import torch                                    # pytorch 
import torch.nn as nn
from torch.utils.data import Dataset, Subset, DataLoader
from torch.cuda.amp import autocast, GradScaler

import PyDRN as prn                             # denoiser model
import subprocess                               # GPU memory check

import matplotlib.pyplot as plt                 # plotting
from astroML.plotting import setup_text_plots
setup_text_plots(fontsize=26, usetex=True)

# import pdb                                      # for debugging 
# pdb.set_trace()

####    content    #####

# Base_CNN (class): base architecture for the residual network at the 1st stage
#
# ResBlock (class): residual block for the model
#
# GWResNet (class): residual convolutional network model
#
# ImageDataset (class): features and targets coupling
#
# DeepTools (class): CNN training for spectrograms classification, loss and accuracy plot, CNN test,
#                    save the model once the CNN training is finished

####  internal function  #########################

def _Conv_norm_kernels(in_channels, out_channels, kernel_size, stride=1, padding=0, bias=True, Gauss_kernel=True):
    """
    Setting convolution operations with xavier initialisation kernels.
    ---------------------------------------------------------------------------
    Ref:
        [1] X. Glorot, Y. Bengio, "Understanding the difficulty of training deep
            feedforward neural networks" (2010)
    """
    
    conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, bias=bias)
    
    if Gauss_kernel:
        nn.init.xavier_normal_(conv.weight)
    else:
        nn.init.xavier_uniform_(conv.weight)
    
    return conv


class Normalize(object):
    """
    Images normalisation wrt mean and std values + normalisation between [0, 1].
    ---------------------------------------------------------------------------
    Ref:
        [1] F. Fleuret, Deep Learning Course 14x050, University of Geneva,
            url: https://fleuret.org/dlc/
    """
    
    def __call__(self, tensor):
        
        norm_tensor = (tensor - torch.mean(tensor))/(torch.std(tensor) + 1e-12)
        norm_tensor = (norm_tensor - torch.min(norm_tensor))/(torch.max(norm_tensor) - torch.min(norm_tensor))

        return norm_tensor

    def __repr__(self):
        return self.__class__.__name__+'()'


def _take_modules(model_attribute):
    """
    List of the modules in the input model attribute.
    """
    
    out_modules = [mod for mod in model_attribute]
    
    return out_modules


def _take_parameters(model_module, to_numpy = True):
    """
    List of the parameters in the input model module.
    """
    
    out_params = [list(mod.parameters()) for mod in model_module]
    
    if to_numpy:
        out_params = [param.detach().cpu().numpy() for param in out_params]
    
    return out_params


def _show_kernels(module_params, channel_kernel=0, save_fig=False, directory=None):
    """
    Plot of the convolutional kernels.
    """
    
    for i in module_params:
        
        plt.figure(None, tight_layout=True)
        
        if len(i.shape) == 4:
            a = plt.imshow(i[channel_kernel, channel_kernel, :, :], cmap='Greys', vmin=-1, vmax=1)
            title = f'Conv. kernel {i.shape}, channel {channel_kernel}'
            
        plt.colorbar(a)
        plt.title(title)
        
        if save_fig:
            if directory is None:
                raise ValueError("specify directory to save the kernels images.")
            
            plt.savefig(directory + title + '.png', bbox_inches='tight', pad_inches=0)
            plt.close()
        
        else:
            plt.show()


class _warm_up_gpu(nn.Module):
    """
    Simple 1D CNN model to warm-up the gpu.
    """
    
    def __init__(self):
        super(_warm_up_gpu, self).__init__()
        
        # set up layers
        self.layers = nn.Sequential(
            nn.Conv1d(in_channels=1, out_channels=8, kernel_size=5, padding=2),
            nn.PReLU(),
            nn.Conv1d(in_channels=8, out_channels=8, kernel_size=5, padding=2),
            nn.PReLU(),
            nn.Conv1d(in_channels=8, out_channels=1, kernel_size=5, padding=2))
    
    def forward(self, x):
        x = self.layers(x)
        
        return x
    
    def warm_up(self):
        model = _warm_up_gpu()
        
        loss_fn = nn.MSELoss()
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
        
        if torch.cuda.is_available():
            device = torch.device("cuda:0")
            print("Training on GPU...")
            model = model.to(device)
        else:
            raise ValueError("GPU not available.")
        
        for i in range(5000):
            inputs = torch.randn(20, 1, 100).to(device)   # (batch_size, input_channels, input_length)
            targets = torch.randn(20, 1, 100).to(device)  # (batch_size, output_channels, output_length)
            
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = loss_fn(outputs, targets)
            loss.backward()
            optimizer.step()
    
        print("GPU warmed up!")
    
####    codes    ###################################


class Base_3CNN(nn.Module):
    """
    Base architecture for the residual convolutional network model.
    ---------------------------------------------------------------------------
    Parameters:
        rgb (bool): choose if input images are RGB or greyscale (default = True)
        act_func (str): select the activation funtion used in the residual
                        block ('ReLU' or 'PReLU', default = PReLU)
        
    Architecture:
        1. 3x: Conv, BatchNorm, PReLU (or ReLU)
    
    Ref:
        [1] K. He et al., "Deep Residual Learning for Image Recognition" (2015)
    """
    
    def __init__(self, rgb=True, act_func='PReLU'):
        super(Base_3CNN, self).__init__()
        
        # set images input and out channel
        if rgb:
            ch = 3
        else:
            ch = 1
        
        # set up convolution with Gaussian kernels
        conv1 = _Conv_norm_kernels(in_channels=ch, out_channels=32, kernel_size=9, padding=4)
        conv2 = _Conv_norm_kernels(in_channels=32, out_channels=32, kernel_size=7, padding=3)      
        
        conv3 = _Conv_norm_kernels(in_channels=32, out_channels=ch, kernel_size=5, padding=2)
        
        # set up first two layers (fixed)
        self.initial_layers = nn.Sequential(
            conv1, nn.BatchNorm2d(num_features=32), self._act_func(act_func),
            conv2, nn.BatchNorm2d(num_features=32), self._act_func(act_func))
        
        # set up 3rd layer (the last of the network for each training stage)
        self.last_layer = nn.Sequential(
            conv3, nn.BatchNorm2d(num_features=ch), self._act_func(act_func))
    
    def _act_func(self, activation):
        
        if activation == 'PReLU':
            act = nn.PReLU()
        elif activation == 'ReLU':
            act = nn.ReLU()
        else:
            raise ValueError("You must choose the activation function for the res. block: 'ReLU' or 'PReLU'")
        
        return act
        


class Base_6CNN(nn.Module):
    """
    Base architecture for the residual convolutional network model.
    ---------------------------------------------------------------------------
    Parameters:
        rgb (bool): choose if input images are RGB or greyscale (default = True)
        act_func (str): select the activation funtion used in the residual
                        block ('ReLU' or 'PReLU', default = PReLU)

    Architecture:
        1. 6x: Conv, BatchNorm, PReLU (or ReLU)
    
    Ref:
        [1] K. He et al., "Deep Residual Learning for Image Recognition" (2015)
    """
    
    def __init__(self, rgb=True, act_func='PReLU'):
        super(Base_6CNN, self).__init__()
        
        # set images input and out channel
        if rgb:
            ch = 3
        else:
            ch = 1
        
        # set up convolution with Gaussian kernels
        conv1 = _Conv_norm_kernels(in_channels=ch, out_channels=32, kernel_size=9, padding=4)
        conv2_1 = _Conv_norm_kernels(in_channels=32, out_channels=32, kernel_size=7, padding=3)
        conv2_2 = _Conv_norm_kernels(in_channels=32, out_channels=64, kernel_size=5, padding=2)
        conv2_3 = _Conv_norm_kernels(in_channels=64, out_channels=64, kernel_size=5, padding=2)
        conv2_4 = _Conv_norm_kernels(in_channels=64, out_channels=32, kernel_size=3, padding=1)
        
        conv3 = _Conv_norm_kernels(in_channels=32, out_channels=ch, kernel_size=5, padding=2)
        
        # set up first two layers (fixed)
        self.initial_layers = nn.Sequential(
            conv1, nn.BatchNorm2d(num_features=32), self._act_func(act_func),
            conv2_1, nn.BatchNorm2d(num_features=32), self._act_func(act_func),
            conv2_2, nn.BatchNorm2d(num_features=64), self._act_func(act_func),
            conv2_3, nn.BatchNorm2d(num_features=64), self._act_func(act_func),
            conv2_4, nn.BatchNorm2d(num_features=32), self._act_func(act_func))
        
        # set up 3rd layer (the last of the network for each training stage)
        self.last_layer = nn.Sequential(
            conv3, nn.BatchNorm2d(num_features=ch), self._act_func(act_func))
    
    def _act_func(self, activation):
        
        if activation == 'PReLU':
            act = nn.PReLU()
        elif activation == 'ReLU':
            act = nn.ReLU()
        else:
            raise ValueError("You must choose the activation function for the res. block: 'ReLU' or 'PReLU'")
        
        return act



class ResBlock(nn.Module):
    """
    Residual Block for the residual convolutional network model.
    ---------------------------------------------------------------------------
    Parameters:
        act_func (str): select the activation funtion used in the residual
                        block ('ReLU' or 'PReLU', default = PReLU)
    
    Architecture:
        1. Conv + Batch norm
        2. ReLU or PReLU (default = PReLU)
        3. Conv
    
    Ref:
        [1] K. He et al., "Deep Residual Learning for Image Recognition" (2015)
    """
    
    def __init__(self, act_func='PReLU'):
        
        super(ResBlock, self).__init__()
        
        # set up activation function
        if act_func == 'PReLU':
            self.activation = nn.PReLU()
        elif act_func == 'ReLU':
            self.activation = nn.ReLU()
        else:
            raise ValueError("You must choose the activation function for the res. block: 'ReLU' or 'PReLU'")
        
        # set up convolution with Gaussian kernels
        res_conv1 = _Conv_norm_kernels(in_channels=32, out_channels=32, kernel_size=5, padding=2)
        res_conv2 = _Conv_norm_kernels(in_channels=32, out_channels=32, kernel_size=5, padding=2)
        
        # set up layers in the res. block
        self.block = nn.Sequential(
            res_conv1, nn.BatchNorm2d(num_features=32), self.activation, res_conv2)
    
    
    def forward(self, x):
        
        out = self.block(x)
        out += x
        
        return out



#### class: CNN model for spectrograms (binary) classification
class GWResNet(nn.Module):
    """
    Residual CNN model for (binary) classification.
    ---------------------------------------------------------------------------
    Parameters:
        num_blocks (int): initial number of residual blocks in the CNN model (default = 0)
        act_func (str): residual blocks activation funtion ('ReLU' or 'PReLU', default = PReLU)
        stoch_depth (bool): if True the stochastic depth method for the residual blocks is
                            activated (default = False)
        prob (int, float): probability for each residual block to be dropped during training
                           (default = 0)
    
    Architecture:
        1. 2x or 5x initial Conv, BatchNorm, PReLU
        
        2. Residual Blocks
        
        3. Conv, BatchNorm, PReLU
    
    Ref:
        [1] K. He et al., "Deep Residual Learning for Image Recognition" (2015)
        [2] H. Ren et al., "DN-ResNet: Efficient Deep Residual Network for Image Denoising" (2018)
        [3] G. Huang et al., "Deep Networks with Stochastic Depth" (2016)
    """

    def __init__(self, num_blocks=0, act_func='PReLU', stoch_depth=False, prob=0):
        
        super(GWResNet, self).__init__()
        
        base = Base_6CNN(rgb=True, act_func='ReLU')
        
        # set up complete architecture        
        self.start = base.initial_layers
        
        # initial number of residual blocks
        if num_blocks > 0:
            self.resblocks = nn.ModuleList([ResBlock(act_func) for _ in range(num_blocks)])
        elif num_blocks == 0:
            self.resblocks = nn.ModuleList([nn.Identity()])
        else:
            raise ValueError("num_blocks must be >= 0 (default = 0).")    
        
        self.end = base.last_layer
        
        # set up stochastic depth for residual blocks
        self.stoch_depth = stoch_depth
        
        if prob >= 0 and prob <= 1:
            self.prob = prob
        else:
            raise ValueError("being a probability, prob must be chosen between [0, 1].")
    
    
    def forward(self, x):
        
        x = self.start(x)
        
        if self.training and self.stoch_depth:
            for rb in self.resblocks:
                if torch.rand(1, device=x.device) < (1 - self.prob):
                    x = rb(x)        
        else:
            for rb in self.resblocks:
                x = rb(x)
        
        x = self.end(x)
        
        return x


#############################################################################################################################


class ImageDataset(Dataset):
    """
    Features and targets coupling.
    ---------------------------------------------------------------------------
    Ref:
        [1] S. Raschka, Y. H. Liu, V. Mirjalili "Machine Learning with PyTorch
            and Scikit-Learn" (2022)
    """
    
    def __init__(self, file_list, labels, transform=None):
        
        self.file_list = file_list
        self.labels = labels
        self.transform = transform
    
    
    def __getitem__(self, index):
        
        img = Image.open(self.file_list[index]).convert("RGB")
        
        if self.transform is not None:
            img = self.transform(img)
        
        label = self.labels[index]
        
        return img, label
    
    
    def __len__(self):
        return len(self.labels)


#############################################################################################################################


# class: tools for cnn
class DeepTools:
    """
    This class contains the functions to train the CNN model, to plot the loss and the
    accuracy of the model, to test the CNN and to save/load the trained model.
    ---------------------------------------------------------------------------
    Attributes:
        model (nn.Module): CNN model for denoising from DnsCNet class (in train_model module)
        loss_fn (nn.Module): loss for the training (Mean Squared Error Loss, in train_model module)
        optimizer (torch.optim): features optimizer (Adam, in train_model module)
        device (torch): device on which the computation is done (in train_model module)

    Methods:
        make_dataset: it defines the train datasets
        train_model: it trains the CNN model
        cascade_training: it performs the cascade learning of the model
        show_model: it shows the loss and the accuracy of the trained CNN model
        test_model: it tests the CNN model after the training
        save_model: it saves the CNN model (or if you want to save a checkpoint during training)
        load_model: it loads the saved CNN model (or the checkpoint to continue the CNN training)
        
    Ref:
        [1] S. Raschka, Y. H. Liu, V. Mirjalili "Machine Learning with PyTorch
            and Scikit-Learn" (2022)
        [2] F. Fleuret, Deep Learning Course 14x050, University of Geneva,
            url: https://fleuret.org/dlc/
        [3] H. Ren et al., "DN-ResNet: Efficient Deep Residual Network for Image Denoising" (2018)
    """
    ##########################################################################################
    def _gpu_memory_nvidia_smi(self):
        try:
            # nvidia-smi command to get GPU memory usage
            result = subprocess.check_output(['nvidia-smi'], encoding='utf-8')
    
            # search for memory usage information in the output
            memory_usage_info = ""
            for line in result.splitlines():
                if "MiB / " in line:
                    memory_usage_info = line.strip()
                    break
    
            if memory_usage_info:
                print("\n GPU Memory Usage:", memory_usage_info)
            else:
                print("\n Unable to find GPU memory usage information.")
        
        except subprocess.CalledProcessError:
            print("\n Error running nvidia-smi command. Make sure it is installed and accessible.")
    
    
    def _quicker_train(self, num_stages, n_times, start_stage, model, epochs, learn_rate, train_dataset, valid_dataset,
                       stage_step=1, act_func='PReLU', stoch_depth=False, prob=0):
        
        stages = [i*num_stages*stage_step for i in range(n_times)]
        
        loss_train, accuracy_train = [], []
        if valid_dataset is not None:
            loss_valid, accuracy_valid = [], []
        else:
            loss_valid, accuracy_valid = None, None
        
        for s in tqdm(stages):
            self.model, tl, ta, vl, va = self.cascade_training(num_stages, model, epochs, learn_rate, train_dataset, valid_dataset,
                                                               stage = s + start_stage, stage_step=stage_step,
                                                               act_func=act_func, stoch_depth=stoch_depth, prob=prob)
            
            loss_train += tl
            accuracy_train += ta
            try:
                loss_valid += vl
                accuracy_valid += va
            except:
                pass
        
        return self.model, loss_train, accuracy_train, loss_valid, accuracy_valid
    ##########################################################################################
    
    
    def make_dataset(self, data_path, batch_size, transform=None, valid_size = None):
        """
        Train dataset generation (and also validation dataset if valid_size is inserted).
        -------------------------------------------------------------------
        Par:
            data_path (str): path for the data
            batch_size (int): batch size for the train (and validation) dataset
            transform (torchvision.transforms): transformation to apply to the images (default = None)
            valid_size (int): validation dataset size (default = None)
            
        Return:
            train_dataset (torch.utils.data.DataLoader): train dataset
            valid_dataset (torch.utils.data.DataLoader): validation dataset (if valid_size is defined)
        """
        
        # load images
        imgdir_path = pathlib.Path(data_path)
        file_list = sorted([str(path) for path in imgdir_path.glob('*.png')])
        random.shuffle(file_list)

        # define labels
        labels = [0 if 'noise' in os.path.basename(file) else 1 for file in file_list]
        
        # create the dataset
        image_dataset = ImageDataset(file_list, labels, transform=transform)
        
        # split and define train and validation dataset
        if valid_size is not None:
            validation = Subset(image_dataset, torch.arange(valid_size))
            training = Subset(image_dataset, torch.arange(valid_size, len(image_dataset)))
            
            train_dataset = DataLoader(training, batch_size, shuffle=True)
            valid_dataset = DataLoader(validation, batch_size, shuffle=False)
            
            return train_dataset, valid_dataset
        
        else:
            train_dataset = DataLoader(image_dataset, batch_size, shuffle=True)
            return train_dataset


    def train_model(self, model, epochs, learn_rate, train_dataset, valid_dataset = None,
                    stage = None, act_func = 'PReLU', stoch_depth = False, prob = 0):
        """
        Training of the CNN model defined in DnsResNet.
        ------------------------------------------------------
        Par:
            model (torch): CNN model
            epochs (int): number of iterations for the model training
            learn_rate (float): learning rate parameter for the model optimization
            train_dataset (torch.utils.data.DataLoader): training dataset
            valid_dataset (torch.utils.data.DataLoader): validation dataset (default = None)
            stage (int): stage of the cascade learning (default = None)
            act_func (str): residual blocks activation funtion ('ReLU' or 'PReLU', default = PReLU)
            stoch_depth (bool): if True the stochastic depth method for the residual blocks is
                                activated after the 10th stage (default = False)
            prob (int, float): probability for each residual block to be dropped during training
                               (default = 0)
        
        Return:
            model (torch): trained model
            mean_loss_train (list): mean loss values for the model training
            mean_accuracy_train (list): mean accuracy values for the model training
            mean_loss_valid (list): mean loss values for the model validation (if valid_dataset is inserted)
            mean_accuracy_valid (list): mean accuracy values for the model validation (if valid_dataset is inserted)
        """
        
        # set model and stage as global variables
        self.model = model
        
        # set cascade learning and add a residual block to the model
        if stage is not None:
            if stage == 0:
                pass
            elif stage == 1:
                self.model.resblocks.append(ResBlock(act_func))
            elif stage >= 2:
                self.model.resblocks.extend([ResBlock(act_func) for _ in range(stage - self.stage)])
            else:
                raise ValueError("stage (int) must be >= 0.")
        
        # set stage
        self.stage = stage
        
        # set up stochastic depth for residual blocks (if there are more than 10 resblocks)
        if stoch_depth:
            self.model.stoch_depth = True
            self.model.prob = prob
        
        # define loss and optimizer
        self.loss_fn = nn.MSELoss()
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=learn_rate)
        # self.optimizer = torch.optim.SGD(self.model.parameters(), lr=learn_rate)
        
        # print loss
        print_every = epochs/30
        
        # control device
        if torch.cuda.is_available():
            self.device = torch.device("cuda:0")
            print("Training on GPU...")
            self.model = self.model.to(self.device)
        
        else:
            print("No GPU available, redirecting to CPU...\n")
            user_input = input("Continue training on CPU? (y/n): ")
            
            if user_input.lower() == "n":
                raise Exception("Training interrupted")
            else:
                self.device = torch.device("cpu")
        
        # define lists for loss and accuracy (both train and validation)
        self.mean_loss_train = [0]*epochs
        self.mean_accuracy_train = [0]*epochs
        
        if valid_dataset is not None:
            self.mean_loss_valid = [0]*epochs
            self.mean_accuracy_valid = [0]*epochs
        else:
            self.mean_loss_valid = None
            self.mean_accuracy_valid = None
        
        # reduce memory cost by mixing the precision of float data
        scaler = GradScaler()
        
        # training loop
        for epoch in tqdm(range(epochs)):
            
            # model training
            self.model.train()
            
            for x_batch, y_batch in train_dataset:
                x_batch = x_batch.to(self.device) 
                y_batch = y_batch.to(self.device)
                
                self.optimizer.zero_grad()                                          # put the optimizer grad to zero
                
                # mixed precision for float                
                with autocast():
                    pred = self.model(x_batch)                                      # model prediction
                    loss = self.loss_fn(pred, y_batch)                              # model loss
                
                scaler.scale(loss).backward()                                       # backward propagation 
                scaler.step(self.optimizer)                                         # model parameters optimization
                scaler.update()
                
                self.mean_loss_train[epoch] += loss.item()*y_batch.size(0)                  # store single loss values 
                is_correct = (torch.argmax(pred, dim=1) == y_batch).float().cpu().numpy()
                self.mean_accuracy_train[epoch] += is_correct.sum()                         # store single accuracy values

            self.mean_loss_train[epoch] /= len(train_dataset.dataset)               # mean loss value for the epoch
            self.mean_accuracy_train[epoch] /= len(train_dataset.dataset)           # mean accuracy value for the epoch
            
            if int(print_every) >= 1 and epoch % int(print_every) == 1:
                print("####################\n",
                      f"Training Loss: {self.mean_loss_train[epoch]:.4f}")
            
            # model validation
            if valid_dataset is not None:
            
                self.model.eval()
                
                with torch.no_grad():
                    for x_batch, y_batch in valid_dataset:
                        x_batch = x_batch.to(self.device) 
                        y_batch = y_batch.to(self.device)
                        
                        valid_pred = self.model(x_batch)                                                 # model prediction for validation
                        valid_loss = self.loss_fn(valid_pred, y_batch)                                   # model loss for validation
                        self.mean_loss_valid[epoch] += valid_loss.item()*y_batch.size(0)                 # store single validation loss values
                        is_correct = (torch.argmax(valid_pred, dim=1) == y_batch).float().cpu().numpy() 
                        self.mean_accuracy_valid[epoch] += is_correct.sum()                              # store single validation accuracy values
    
                self.mean_loss_valid[epoch] /= len(valid_dataset.dataset)                  # validation mean loss value for the epoch
                self.mean_accuracy_valid[epoch] /= len(valid_dataset.dataset)              # validation mean accuracy value for the epoch
                
                if int(print_every) >= 1 and epoch % int(print_every) == 1:
                    print(f"Validation Loss: {self.mean_loss_valid[epoch]:.4f}")
                
        # return output
        return self.model, self.mean_loss_train, self.mean_accuracy_train, self.mean_loss_valid, self.mean_accuracy_valid
    
    
    def cascade_training(self, num_stages, model, epochs, learn_rate, train_dataset, valid_dataset = None,
                         stage = 0, stage_step = 1, act_func = 'PReLU', stoch_depth = False, prob = 0):
        """
        Cascade learning of the CNN model defined in DnsResNet.
        ------------------------------------------------------
        Par:
            num_stages (int): number of stages of the training
            model (torch): CNN model
            epochs (int): number of iterations for the model training
            learn_rate (float): learning rate parameter for the model optimization
            train_dataset (torch.utils.data.DataLoader): training dataset
            valid_dataset (torch.utils.data.DataLoader): validation dataset (default = None)
            stage (int): stage of the cascade learning (default = 0)
            stage_step (int): step in the training stage (default = 1)
            act_func (str): residual blocks activation funtion ('ReLU' or 'PReLU', default = PReLU)
            stoch_depth (bool): if True the stochastic depth method for the residual blocks is
                                activated after the 10th stage (default = False)
            prob (int, float): probability for each residual block to be dropped during training
                               (default = 0)
        
        Return:
            model (torch): trained model
            mean_loss_train (list): mean loss values for the model training
            mean_accuracy_train (list): mean accuracy values for the model training
            mean_loss_valid (list): mean loss values for the model validation (if valid_dataset is inserted)
            mean_accuracy_valid (list): mean accuracy values for the model validation (if valid_dataset is inserted)
        """
        
        # define lists for train loss and accuracy
        tot_loss_train, tot_accuracy_train = [], []
        
        if valid_dataset is not None:
            tot_loss_valid, tot_accuracy_valid = [], []
        else:
            tot_loss_valid, tot_accuracy_valid = None, None
        
        
        for _ in tqdm(range(num_stages)):
            
            # model training
            model, tl, ta, vl, va = self.train_model(model, epochs, learn_rate, train_dataset,
                                                     valid_dataset, stage, act_func, stoch_depth, prob)
            
            # stage update
            stage += stage_step
            
            # storing train and valid losses
            tot_loss_train += tl
            tot_accuracy_train += ta
            try:
                tot_loss_valid += vl
                tot_accuracy_valid += va
            except:
                pass
            
            # check on loss values
            if np.isnan(np.sum(tl)) or np.isnan(np.sum(ta)):
                print(f"stage {stage}: in the mean_loss_train list a nan value is present.")
                return self.model, tot_loss_train, tot_accuracy_train, tot_loss_valid, tot_accuracy_valid
            elif valid_dataset is not None and np.isnan(np.sum(vl)) or np.isnan(np.sum(va)):
                print(f"stage {stage}: in the mean_loss_valid list a nan value is present.")
                return self.model, tot_loss_train, tot_accuracy_train, tot_loss_valid, tot_accuracy_valid
            else:
                pass
            
            # check on GPU memory
            self._gpu_memory_nvidia_smi()
        
        return model, tot_loss_train, tot_accuracy_train, tot_loss_valid, tot_accuracy_valid
    

    def show_model(self, comp_dloss=False, stage=None, train_loss=None,
                   train_accuracy=None, valid_loss=None, valid_accuracy=None):
        """
        Plots of the trained CNN model loss and accuracy (also with validation if
        valid_dataset in train_model() is defined).
        ------------------------------------------------------
        Par:
            comp_dloss (bool): if True computes an approximate derivative for the train loss (default = False)
            stage (int): the stage in the cascade learning can be specified in the title (default = None)
            train_loss (list, array): show a specific train loss (default = None)
            train_accuracy (list, array): show a specific train accuracy (default = None)
            valid_loss (list, array): show a specific validation loss (default = None)
            valid_accuracy (list, array): show a specific validation accuracy (default = None)
        """
        
        # define losses
        if train_loss is not None:
            tl = train_loss
            ta = train_accuracy
            vl = valid_loss
            va = valid_accuracy
        else:
            tl = self.mean_loss_train
            ta = self.mean_accuracy_train
            vl = self.mean_loss_valid
            va = self.mean_accuracy_valid
        
        # define ascissa
        x_arr = np.arange(len(tl)) + 1
        
        # define plot title
        title_loss = 'Model mean loss'
        title_accuracy = 'Model mean accuracy'
        if stage is not None and isinstance(stage, int):
            title_loss += ': stage ' + str(stage)
            title_accuracy += ': stage ' + str(stage)
        else:
            raise ValueError("stage must be an integer.")
        
        # loss plot
        fig = plt.figure(num = None, figsize = (12, 12), tight_layout = True)
        ax = fig.add_subplot(111)
        ax.plot(x_arr, tl, label='train loss')
        
        if vl is not None:
            ax.scatter(x_arr, vl, label='valid. loss')
        
        plt.xlim((0, len(tl) + 1))
        plt.ylim((np.mean(tl) - np.std(tl),
                  np.mean(tl) + np.std(tl)))
        
        plt.xlabel('epoch')
        plt.ylabel('loss')
        plt.title(title_loss)            
        plt.legend(loc = 'best')
        ax.grid(True)
        ax.label_outer()            
        ax.tick_params(which='both', direction='in',width=2)
        ax.tick_params(which='major', direction='in',length=7)
        ax.tick_params(which='minor', direction='in',length=4)
        ax.xaxis.set_ticks_position('both')
        ax.yaxis.set_ticks_position('both')
        plt.show()
        
        # accuracy plot
        fig = plt.figure(num = None, figsize = (12, 12), tight_layout = True)
        ax = fig.add_subplot(111)
        ax.plot(x_arr, ta, label='train acc.')
        
        if va is not None:
            ax.scatter(x_arr, va, label='valid. acc.')
        
        plt.xlim((0, len(tl) + 1))
        plt.ylim((0, 1))
        plt.xlabel('epoch')
        plt.ylabel('accuracy')
        plt.title(title_accuracy)
        plt.legend(loc = 'best')
        ax.grid(True)
        ax.label_outer()            
        ax.tick_params(which='both', direction='in',width=2)
        ax.tick_params(which='major', direction='in',length=7)
        ax.tick_params(which='minor', direction='in',length=4)
        ax.xaxis.set_ticks_position('both')
        ax.yaxis.set_ticks_position('both')
        plt.show()
        
        # loss derivative
        if comp_dloss:
            dloss = [tl[l + 1] - tl[l]
                     for l in range(len(tl) - 1)]
            
            fig = plt.figure(num = None, figsize = (12, 12), tight_layout = True)
            ax = fig.add_subplot(111)
            ax.plot(x_arr[:-1], dloss)
            plt.xlim((0, len(dloss)))
            plt.ylim((np.mean(dloss) - np.std(dloss), np.mean(dloss) + np.std(dloss)))
            plt.xlabel('epoch')
            plt.ylabel('loss change rate')
            plt.title('Model loss change rate')
            ax.grid(True)
            ax.label_outer()            
            ax.tick_params(which='both', direction='in',width=2)
            ax.tick_params(which='major', direction='in',length=7)
            ax.tick_params(which='minor', direction='in',length=4)
            ax.xaxis.set_ticks_position('both')
            ax.yaxis.set_ticks_position('both')
            plt.show()
    
    
    def test_model(self, test_dataset, model = None):
        """
        Test of the CNN model after the training.
        ------------------------------------------------------
        Par:
            test_dataset (torch.utils.data.DataLoader): test dataset
            model (torch): CNN model (if None, the used model is the one
                           in the train_model module, default = None)
        """
        
        # waits for all kernels in all streams on a CUDA device to complete 
        torch.cuda.synchronize()
        
        # to cpu (for the test)
        if model is None:
            model = self.model.cpu()
        
        # initialize test loss
        self.mean_accuracy_test = 0
        
        # test CNN
        model.eval()
        
        with torch.no_grad():
            for x_batch, y_batch in test_dataset:
                test_pred = model(x_batch)
                is_correct = (torch.argmax(test_pred, dim=1) == y_batch).float().numpy()
                self.mean_accuracy_test += is_correct.sum()
        
        # test mean accuracy value
        self.mean_accuracy_test /= len(test_dataset.dataset)
        print('The mean accuracy value for the test dataset is:', self.mean_accuracy_test)
    
    
    def save_model(self, batch_size, epochs, learning_rate, stages, activation, filename, denoiser=None, train_loss=None,
                   train_accuracy=None, valid_loss=None, valid_accuracy=None, notes=None, gpu=None, start_resblocks=[0, 'PReLU']):
        """
        To save the CNN model after training (or a checkpoint during training); this module saves the model
        by creating a dictionary in which the model features are stored, such as the model, the model state,
        the epochs, the train and validation (if inserted) mean losses and mean accuracy, the device, the
        optimizer state and the batch size.
        ------------------------------------------------------
        Par:
            batch_size (int): batch size of the training dataset for the training process
            epochs (int): number of iterations for the model training
            learning_rate (float): learning rate parameter for the model optimization
            stages (int): number of stage in the cascade learning
            activation (str): activation function in the residual blocks
            filename (str): name of the CNN model (.pt or .pth, the filepath where to save the
                            model is defined inside the module)
            denoiser (str): name of the denoiser used in the CNN (default = None)
            train_loss (list, array): save a specific train loss (default = None)
            train_accuracy (list, array): save a specific train accuracy (default = None)
            valid_loss (list, array): save a specific validation loss (default = None)
            valid_accuracy (list, array): save a specific validation accuracy (default = None)
            notes (str): possible notes for model description (default = None)
            gpu (str): GPU memory used (default = None)
            start_resblocks (list): initial residual blocks in the model (default = [0, 'PReLU'])
        """
        
        filepath = "D:/Home/Universita'/Universita'/Magistrale/Master Thesis/Tesi_Codes/Prova_cnn/CNN_models/GWResNet_Models/"
        
        # define losses and accuracy
        if train_loss is not None:
            tl, ta = train_loss, train_accuracy
            vl, va = valid_loss, valid_accuracy
        else:
            tl, ta = self.mean_loss_train, self.mean_accuracy_train
            vl, va = self.mean_loss_valid, self.mean_accuracy_valid
        
        checkpoint = {'model': self.model,
                      'state_dict': self.model.state_dict(),
                      'epochs': epochs,
                      'learning_rate': learning_rate,
                      'mean_loss_train': tl,
                      'mean_accuracy_train': ta,
                      'mean_loss_valid': vl,
                      'mean_accuracy_valid': va,
                      'device': self.device,
                      'denoiser': denoiser,
                      'optimizer_state': self.optimizer.state_dict(),
                      'batch_size': batch_size,
                      'stages': stages,
                      'activation': activation,
                      'notes': notes,
                      'gpu': gpu,
                      'start_resblocks': start_resblocks}
        
        try:
            checkpoint['mean_loss_test'] = self.mean_loss_test
        except:
            pass
        
        torch.save(checkpoint, filepath + filename)
    
    
    def load_model(self, filename):
        """
        To load the CNN model (or a checkpoint to continue the CNN training).
        ------------------------------------------------------
        Par:
            filename (str): name of the CNN model (.pt or .pth, the filepath where to save the
                            model is defined inside the module)
            
        Return:
            model (torch object): CNN model
            obs (dict): dictionary with epochs, loss, accuracy and batch size
        """
        
        filepath = "D:/Home/Universita'/Universita'/Magistrale/Master Thesis/Tesi_Codes/Prova_Denoiser/DNS_models/ResNet_models/"
        
        # load the model
        checkpoint = torch.load(filepath + filename)
        
        # define update model
        model = checkpoint['model']
        model.optimizer_state = checkpoint['optimizer_state']
        model.load_state_dict(checkpoint['state_dict'])
        model.device = checkpoint['device']
        model.average_loss = checkpoint['mean_loss_train']
        
        # dict with other info
        obs = {'epochs': checkpoint['epochs'],
               'learning_rate': checkpoint['learning_rate'],
               'batch_size': checkpoint['batch_size'],
               'stages': checkpoint['stages'],
               'activation': checkpoint['activation'],
               'notes': checkpoint['notes'],
               'gpu': checkpoint['gpu'],
               'start_resblocks': checkpoint['start_resblocks'],
               'mean_loss_train': checkpoint['mean_loss_train'],
               'mean_loss_valid': checkpoint['mean_loss_valid'],
               'mean_accuracy_train': checkpoint['mean_accuracy_train'],
               'mean_accuracy_valid': checkpoint['mean_accuracy_valid']}
        
        try:
            obs['mean_loss_test'] = checkpoint['mean_loss_test']
        except:
            pass
        
        return model, obs



# end
