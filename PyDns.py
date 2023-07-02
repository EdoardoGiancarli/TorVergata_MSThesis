# -*- coding: utf-8 -*-
"""
Created on Thu Jun 29 21:38:44 2023

@author: Edoardo Giancarli
"""

#### ModuleMSThesis - Spectrograms Denoising version 2 ###############################################

####   libraries   #####

import numpy as np                              # operations
import pandas as pd                             # dataframe 
import pathlib                                  # filepaths
import os
from PIL import Image                           # images

from tqdm import tqdm                           # loop progress bar
import cv2                                      # images visualization

import PySim as psm                             # signal simulation and injection
import PySpectr as psr                          # signals spectrograms

import torch                                    # pytorch 
import torch.nn as nn
from torch.utils.data import Dataset, Subset, DataLoader
from torch.cuda.amp import autocast, GradScaler

import matplotlib.pyplot as plt                 # plotting
from astroML.plotting import setup_text_plots
setup_text_plots(fontsize=26, usetex=True)

####    content    #####

# rename_noise_imgs (function): rename noise images
#
# cv2disp (function): display the images
# 
# img_generator (function): generate clean and noisy image
#
# DnsCNet (class): cnn model for denoising
#
# ImageDataset (class): dataset setting
#
# DnsCNetTools (class): tools for cnn

####    codes    #####

# rename noise images
def rename_noise_imgs(directory_path, prefix):
    
    # get a list of all files in the directory
    file_list = os.listdir(directory_path)

    # iterate over each file
    for filename in file_list:
        if 'noise' in filename:
            
            # create the new filename by prepending the prefix
            new_filename = prefix + filename
    
            # build the full file paths
            current_path = os.path.join(directory_path, filename)
            new_path = os.path.join(directory_path, new_filename)
    
            # rename the file
            os.rename(current_path, new_path)


# display the images
def cv2disp(img_name, img, x_pos = 10, y_pos = 10):
    
    cv2.imshow(img_name, img/(np.max(img) + 1e-10))
    cv2.moveWindow(img_name, x_pos, y_pos)


# generate clean and noisy image
def img_generator(N, clean_imgs_path, noisy_imgs_path, lfft, max_distr = True, save_max_distr = True):
    
    # path bsd with interf data
    path_bsd_gout = "D:/Home/Universita'/Universita'/Magistrale/Master Thesis/Tesi_Codes/gout_58633_58638_295_300.mat"
    
    # initialize N of computed spectrograms
    s = 0
    
    # initialize list for computed spectrograms maxima
    max_list = []
    
    for n in tqdm(range(N)):
        
        print('############################################\n')
                
        # initialize random parameters for the long transient signals
        fgw0 = 301.01 + np.random.uniform(0, 20)
        tcoe = 58633 + np.random.uniform(0, 4)
        tau = 1 + np.random.uniform(0, 5)
        eta = np.random.uniform(-1, 1)
        psi = np.random.uniform(0, 90)
        right_asc = np.random.uniform(0, 360)
        dec = np.random.uniform(-90, 90)
        
        # gw parameters
        params = psm.parameters(days = 3, dt = 0.5, fgw0 = fgw0, tcoe = tcoe, n = 5, k = None, tau = tau, Phi0 = 0,
                                right_asc = right_asc, dec = dec, eta = eta, psi = psi, Einstein_delay = False,
                                Doppler_effect = True, interferometer = 'LIGO-L', h0factor=1e22, signal_inj = True,
                                bsd_gout = path_bsd_gout, key='gout_58633_58638_295_300', mat_v73=True)
        
        # signal simulation and injection into noise
        gwinj_clean = psm.GW_injection(params, amp = 5e1)
        gwinj_noisy = psm.GW_injection(params, amp = 1e-1)
        
        try:
            y_clean = gwinj_clean.injection()
            y_noisy = gwinj_noisy.injection()
            
            # loop for the spectrograms (if there are too many zeros it skips the chunk)
            if len(np.where(y_clean == 0)[0])/len(y_clean) < 0.1:
                
                # spectrograms
                m1 = psr.spectr(y_clean, params['dt'], lfft, title = 'clean_spectr' + str(s),
                                images = True, directory = clean_imgs_path)
                
                psr.spectr(y_noisy, params['dt'], lfft, title = 'noisy_spectr' + str(s),
                           images = True, directory = noisy_imgs_path)
                
                # update number of computed spectrogram
                s += 1
                
                # update max_list
                if max_distr:
                    max_list.append(m1)
                
            else:
                pass
        except ValueError:
            pass
        except IndexError:
            pass
    
    if max_distr:
        if save_max_distr:
            df_path = "D:/Home/Universita'/Universita'/Magistrale/Master Thesis/Tesi_Codes/Max_Spectr_Distr/"
            pd.DataFrame(max_list, columns=['max spectrograms']).to_csv(df_path + 'Distr_max_clear_imgs_DnsCNet.csv')
        
        return max_list
    else:
        pass


#############################################################################################################################
# class: cnn model for denoising
class DnsCNet(nn.Module):
    """
    CNN model for spectrograms denoising.
    ---------------------------------------------------------------------------
    Layers:
        1. Convolutional (with xavier Gaussian kernels) + PReLU activ. func. + Batch Norm
        2. Convolutional + PReLU activ. func.
        3. Convolutional + PReLU activ. func. + Dropout
        4. Convolutional + PReLU activ. func. + Dropout
        5. Convolutional + PReLU activ. func.
    
    Ref:
        [1] Andrew Reader, url: https://www.youtube.com/watch?v=Wq8mh3Y0JjA&t=1759s
        [2] X. Glorot, Y. Bengio, "Understanding the difficulty of training deep
            feedforward neural networks" (2010)
    """
    
    def __init__(self):
        super(DnsCNet, self).__init__()
        
        conv1 = nn.Conv2d(in_channels=1, out_channels=8, kernel_size=7, padding=3)
        nn.init.xavier_normal_(conv1.weight)
        
        self.layers = nn.Sequential(
            conv1, nn.PReLU(), nn.BatchNorm2d(num_features=8),
            nn.Conv2d(in_channels=8, out_channels=16, kernel_size=7, padding=3), nn.PReLU(),
            nn.Conv2d(in_channels=16, out_channels=32, kernel_size=7, padding=3), nn.PReLU(), nn.Dropout2d(p=0.5),
            nn.Conv2d(in_channels=32, out_channels=8, kernel_size=7, padding=3), nn.PReLU(), nn.Dropout2d(p=0.5),
            nn.Conv2d(in_channels=8, out_channels=1, kernel_size=7, padding=3), nn.PReLU())
    
    
    def forward(self, x):
        x = self.layers(x)
        return x


#############################################################################################################################
# class: dataset setting
class ImageDataset(Dataset):
    """
    Features and targets coupling.
    ---------------------------------------------------------------------------
    Ref:
        [1] S. Raschka, Y. H. Liu, V. Mirjalili "Machine Learning with PyTorch
            and Scikit-Learn" (2022)
    """
    
    def __init__(self, in_img_list, trg_list, transform=None):
        
        self.in_img_list = in_img_list
        self.trg_list = trg_list
        self.transform = transform
    
    
    def __getitem__(self, index):
        
        in_img = Image.open(self.in_img_list[index])
        trg_img = Image.open(self.trg_list[index])
        
        if self.transform is not None:
            in_img = self.transform(in_img)
            trg_img = self.transform(trg_img)
                
        return in_img, trg_img
    
    
    def __len__(self):
        return len(self.trg_list)


#############################################################################################################################
# class: tools for cnn
class DnsCNetTools:
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
        show_model: it shows the loss and the accuracy of the trained CNN model
        test_model: it tests the CNN model after the training
        save_model: it saves the CNN model (or if you want to save a checkpoint during training)
        load_model: it loads the saved CNN model (or the checkpoint to continue the CNN training)
        
    Ref:
        [1] S. Raschka, Y. H. Liu, V. Mirjalili "Machine Learning with PyTorch
            and Scikit-Learn" (2022)
    """
    
    def make_dataset(self, input_img_path, target_path, batch_size,
                     transform=None, valid_size = None):
        """
        Train dataset generation (and also validation dataset if valid_size is inserted).
        This module can be also used for the test dataset generation (with valid_size = None).
        -------------------------------------------------------------------
        Par:
            input_img_path (str): path for the input "noisy" images
            target_path (str): path for the target images
            batch_size (int): batch size for the train (and validation) dataset
            transform (torchvision.transforms): transformation to apply to the images
            valid_size (int): validation dataset size (default = None)
            
        Return:
            train_dataset (torch.utils.data.DataLoader): train dataset
            valid_dataset (torch.utils.data.DataLoader): validation dataset (if valid_size is defined)
        """
        
        # load images
        in_imgdir_path = pathlib.Path(input_img_path)
        in_img_list = sorted([str(path) for path in in_imgdir_path.glob('*.png')])
        
        trg_dir_path = pathlib.Path(target_path)
        trg_list = sorted([str(path) for path in trg_dir_path.glob('*.png')])
        
        # create the dataset
        image_dataset = ImageDataset(in_img_list, trg_list, transform=transform)
        
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
    
    
    def train_model(self, model, epochs, learn_rate, train_dataset, valid_dataset = None):
        """
        Training of the CNN model defined in GW_Deep.
        ------------------------------------------------------
        Par:
            model (torch): CNN model
            epochs (int): number of iterations for the model training
            learn_rate (float): learning rate parameter for the model optimization
            train_dataset (torch.utils.data.DataLoader): training dataset
            valid_dataset (torch.utils.data.DataLoader): validation dataset (default = None)

        Return:
            mean_loss_train (list): mean loss values for the model training
            mean_loss_valid (list): mean loss values for the model validation (if valid_dataset is inserted)
        """
        
        # set model as global variable
        self.model = model
        
        # define loss and optimizer
        self.loss_fn = nn.MSELoss()
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=learn_rate)
        
        # print loss
        print_every = epochs//30
        
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
        
        # define lists for train loss
        self.mean_loss_train = [0]*epochs
        
        if valid_dataset is not None:
            self.mean_loss_valid = [0]*epochs
        else:
            self.mean_loss_valid = None
        
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
                
                self.mean_loss_train[epoch] += loss.item()*y_batch.size(0)          # store single loss values 
                
            self.mean_loss_train[epoch] /= len(train_dataset.dataset)               # mean loss value for the epoch
            
            if epoch % print_every == 1:
                print("####################\n",
                      f"Training Loss: {self.mean_loss_train[epoch]:.4f}")
            
            # model validation
            if valid_dataset is not None:
            
                self.model.eval()
                
                with torch.no_grad():
                    for x_batch, y_batch in valid_dataset:
                        x_batch = x_batch.to(self.device) 
                        y_batch = y_batch.to(self.device)
                        
                        valid_pred = self.model(x_batch)                                   # model prediction for validation
                        valid_loss = self.loss_fn(valid_pred, y_batch)                     # model loss for validation
                        
                        self.mean_loss_valid[epoch] += valid_loss.item()*y_batch.size(0)   # store single validation loss values
                            
                self.mean_loss_valid[epoch] /= len(valid_dataset.dataset)                  # validation mean loss value for the epoch
                
                if epoch % print_every == 1:
                    print(f"Validation Loss: {self.mean_loss_valid[epoch]:.4f}")
                
            else:
                pass
            
        if valid_dataset is not None:            
            return self.mean_loss_train, self.mean_loss_valid
        
        else:
            return self.mean_loss_train
    
    
    def show_model(self, comp_dloss = False):
        """
        Plots of the trained CNN model loss and accuracy (also with validation if
        valid_dataset in train_model() is defined).
        ------------------------------------------------------
        Par:
            comp_dloss (bool): if True computes an approximate derivative for the loss (default = None)
        
        Return:
            dloss (list): loss derivative values for the model training (if comp_dloss is True)
        """
        
        # define ascissa
        x_arr = np.arange(len(self.mean_loss_train)) + 1
        
        # loss plot
        fig = plt.figure(num = 1, figsize = (12, 12), tight_layout = True)
        ax = fig.add_subplot(111)
        ax.plot(x_arr, self.mean_loss_train, c = 'OrangeRed', label='train loss')
        
        if self.mean_loss_valid is not None:
            ax.plot(x_arr, self.mean_loss_valid, c = 'LawnGreen', label='valid. loss')
        else:
            pass
        
        plt.xlim((0, len(self.mean_loss_train) + 1))
        plt.ylim((np.mean(self.mean_loss_train) - np.std(self.mean_loss_train),
                  np.mean(self.mean_loss_train) + np.std(self.mean_loss_train)))
        
        plt.xlabel('epoch')
        plt.ylabel('loss')
        plt.title('Model mean loss')
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
            dloss = [self.mean_loss_train[l + 1] - self.mean_loss_train[l]
                     for l in range(len(self.mean_loss_train) - 1)]
            
            fig = plt.figure(num = 2, figsize = (12, 12), tight_layout = True)
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
        else:
            pass
        
        # initialize test loss
        self.mean_loss_test = 0
        
        # test CNN
        model.eval()
        
        with torch.no_grad():
            for x_batch, y_batch in test_dataset:
                
                test_pred = model(x_batch)                                # model prediction for test
                test_loss = self.loss_fn(test_pred, y_batch)              # model loss for test
                
                self.mean_loss_test += test_loss.item()*y_batch.size(0)   # store test loss values
        
        # test mean loss value
        self.mean_loss_test /= len(test_dataset.dataset)                  
        print('The mean loss value for the test dataset is:', self.mean_loss_test)
    
    
    def save_model(self, batch_size, epochs, learning_rate, filename):
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
            filename (str): name of the CNN model (.pt or .pth, the filepath where to save the
                            model is defined inside the module)
        """
        
        filepath = "D:/Home/Universita'/Universita'/Magistrale/Master Thesis/Tesi_Codes/Prova_Denoiser/DNS_models/"
        
        checkpoint = {'model': self.model,
                      'state_dict': self.model.state_dict(),
                      'epochs': epochs,
                      'learning_rate': learning_rate,
                      'mean_loss_train': self.mean_loss_train,
                      'mean_loss_valid': self.mean_loss_valid,
                      'device': self.device,
                      'optimizer_state': self.optimizer.state_dict(),
                      'batch_size': batch_size}
        
        try:
            checkpoint['mean_loss_test'] = self.mean_loss_test
        except:
            pass

        torch.save(checkpoint, filepath + filename)
    
    
    def load_model(self, filepath):
        """
        To load the CNN model (or a checkpoint to continue the CNN training).
        ------------------------------------------------------
        Par:
            filepath (str): path where to save the CNN model (.pt or .pth)
            
        Return:
            model (torch object): CNN model
            obs (dict): dictionary with epochs, loss, accuracy and batch size
        """
        
        # load the model
        checkpoint = torch.load(filepath)
        
        # define update model
        model = checkpoint['model']
        model.optimizer_state = checkpoint['optimizer_state']
        model.load_state_dict(checkpoint['state_dict'])
        model.device = checkpoint['device']
        model.average_loss = checkpoint['mean_loss_train']
        
        # dict with other info
        obs = {'epochs': checkpoint['epochs'],
               'learning_rate': checkpoint['learning_rate'],
               'mean_loss_train': checkpoint['mean_loss_train'],
               'mean_loss_valid': checkpoint['mean_loss_valid'],
               'batch_size': checkpoint['batch_size']}
        
        try:
            obs['mean_loss_test'] = checkpoint['mean_loss_test']
        except:
            pass
        
        return model, obs


# end