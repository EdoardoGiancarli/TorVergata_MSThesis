# -*- coding: utf-8 -*-
"""
Created on Sat May 13 20:27:28 2023

@author: Edoardo Giancarli
"""

#### ModuleMSThesis - CNN for spectrograms classification version 1 ###############################################

####   libraries   #####

import numpy as np                                # operations
import random
import pathlib                                    # filepaths
import os                    
from PIL import Image                             # images
from tqdm import tqdm                             # loop progress bar

import torch                                      # pytorch 
import torch.nn as nn
from torch.utils.data import Dataset, Subset, DataLoader
from torch.cuda.amp import autocast, GradScaler

import matplotlib.pyplot as plt                   # plotting
from astroML.plotting import setup_text_plots
setup_text_plots(fontsize=26, usetex=True)

# import pdb                                      # for debugging 
# pdb.set_trace()

####    content    #####

# GW_Deep (class): CNN model for spectrograms (binary) classification
#
# ImageDataset (class): features and targets coupling
#
# DeepTools (class): CNN training for spectrograms classification, loss and accuracy plot, CNN test,
#                    save the model once the CNN training is finished

####    codes    #####

#### class: CNN model for spectrograms (binary) classification
class GW_Deep(nn.Module):
    """
    CNN model for (binary) classification.
    ---------------------------------------------------------------------------
    Layers:
        
    
    Ref:
        [1] S. Raschka, Y. H. Liu, V. Mirjalili "Machine Learning with PyTorch
            and Scikit-Learn" (2022)
        [2] F. Fleuret, Deep Learning Course 14x050, University of Geneva,
            url: https://fleuret.org/dlc/
    """

    def __init__(self):                    
        super(GW_Deep, self).__init__()
        
        # 1st layer (convolutional)
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=16, kernel_size=5, padding=2)
        self.ReLU1 = nn.ReLU()
        self.bn1 = nn.BatchNorm2d(num_features=16)
        self.pool1 = nn.MaxPool2d(kernel_size=2)
        
        # 2nd layer (convolutional)
        self.conv2 = nn.Conv2d(in_channels=16, out_channels=32, kernel_size=5, padding=2)
        self.ReLU2 = nn.ReLU()
        self.bn2 = nn.BatchNorm2d(num_features=32)
        self.pool2 = nn.MaxPool2d(kernel_size=2)
        
        # 3rd layer (convolutional)
        self.conv3 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=5, padding=2)
        self.ReLU3 = nn.ReLU()
        self.pool3 = nn.MaxPool2d(kernel_size=2)
        
        # flatten operation
        self.flat = nn.Flatten()
        
        # 4th layer (linear)
        self.linear4 = nn.Linear(in_features=16384, out_features=1024)  # in_features = channels x image_h/4 x image_w/4
        self.ReLU4 = nn.ReLU()
        self.bn4 = nn.BatchNorm1d(num_features=1024)
        
        # 5th layer (linear, for output)
        self.linear5 = nn.Linear(in_features=1024, out_features=2)
    
        
    def forward(self, x):
        
        # 1st layer (convolutional)
        x = self.conv1(x)
        x = self.ReLU1(x)
        x = self.bn1(x)
        x = self.pool1(x)
        
        # 2nd layer (convolutional)
        x = self.conv2(x)
        x = self.ReLU2(x)
        x = self.bn2(x)
        x = self.pool2(x)
        
        # 3rd layer (convolutional)
        x = self.conv3(x)
        x = self.ReLU3(x)
        x = self.pool3(x)
        
        # flat the output from the conv layers
        x = self.flat(x)
        
        # 4th layer (linear)
        x = self.linear4(x)
        x = self.ReLU4(x)
        x = self.bn4(x)
        
        # 5th layer (linear, for output)
        x = self.linear5(x)
        
        return x


#############################################################################################################################

#### class: dataset setting
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
        
        img = Image.open(self.file_list[index])
        
        if self.transform is not None:
            img = self.transform(img)
        
        label = self.labels[index]
        
        return img, label
    
    
    def __len__(self):
        return len(self.labels)


#############################################################################################################################

#### class: CNN model investigation
class DeepTools:
    """
    This class contains the functions to train the CNN model, to plot the loss and the
    accuracy of the model, to test the CNN and to save/load the trained model.
    ---------------------------------------------------------------------------
    Attributes:
        model (nn.Module): CNN model from GW_Deep class
        loss_fn (nn.Module): loss for the training (Cross Entropy Loss, in train_model module)
        optimizer (torch.optim): features optimizer (Adam, in train_model module)
        device (torch): device on which the computation is done (in train_model module)

    Methods:
        make_dataset: it defines the train and validation datasets
        train_model: it trains the CNN model defined in GW_Deep
        show_model: it shows the loss and the accuracy of the trained CNN model
        test_model: it tests the CNN model after the training
        save_model: it saves the CNN model (or if you want to save a checkpoint during training)
        load_model: it loads the saved CNN model (or the checkpoint to continue the CNN training)
    
    Ref:
        [1] S. Raschka, Y. H. Liu, V. Mirjalili "Machine Learning with PyTorch
            and Scikit-Learn" (2022)
    """    
    
    def make_dataset(self, data_path, batch_size, transform=None, valid_size = None):
        """
        Train dataset generation (and also validation dataset if valid_size is inserted).
        -------------------------------------------------------------------
        Par:
            data_path (str): path for the data
            batch_size (int): batch size for the train (and validation) dataset
            transform (torchvision.transforms): transformation to apply to the images (default = None)
            valid_size (int): validation dataset size (default = None)
            
        return:
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
            
        return:
            mean_loss_train (list): mean loss values for the model training
            mean_accuracy_train (list): mean accuracy values for the model training
            mean_loss_valid (list): mean loss values for the model validation (if valid_dataset is inserted)
            mean_accuracy_valid (list): mean accuracy values for the model validation (if valid_dataset is inserted)
        """
        
        # set model as global variable
        self.model = model
        
        # define loss and optimizer
        self.loss_fn = nn.CrossEntropyLoss()
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=learn_rate)
        
        # print loss and accuracy
        print_every = epochs//10
        
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
                # loss.backward()
                # self.optimizer.step()
                # self.optimizer.zero_grad()
                
                self.mean_loss_train[epoch] += loss.item()*y_batch.size(0)                  # store single loss values 
                is_correct = (torch.argmax(pred, dim=1) == y_batch).float().cpu().numpy()
                self.mean_accuracy_train[epoch] += is_correct.sum()                         # store single accuracy values

            self.mean_loss_train[epoch] /= len(train_dataset.dataset)               # mean loss value for the epoch
            self.mean_accuracy_train[epoch] /= len(train_dataset.dataset)           # mean accuracy value for the epoch
            
            if epoch % print_every == 1:
                print("####################\n",
                      f"Training Loss: {self.mean_loss_train[epoch]:.4f}, Training Accuracy: {self.mean_accuracy_train[epoch]:.4f}")
            
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
                
                if epoch % print_every == 1:
                    print(f"Validation Loss: {self.mean_loss_valid[epoch]:.4f}, Validation Accuracy: {self.mean_accuracy_valid[epoch]:.4f}")
                
            else:
                pass
        
        if valid_dataset is not None:            
            return self.mean_loss_train, self.mean_accuracy_train, self.mean_loss_valid, self.mean_accuracy_valid
        
        else:
            return self.mean_loss_train, self.mean_accuracy_train
    
    
    def show_model(self):
        """
        Plots of the trained CNN model loss and accuracy (also with validation if
        valid_dataset in train_model() is defined).
        """
        
        # define ascissa
        x_arr = np.arange(len(self.mean_loss_train)) + 1
        
        # loss plot
        fig = plt.figure(num = 1, figsize = (12, 12), tight_layout = True)
        ax = fig.add_subplot(111)
        ax.plot(x_arr, self.mean_loss_train, '-o', label='train loss', markersize=12)
        
        if self.mean_loss_valid is not None:
            ax.plot(x_arr, self.mean_loss_valid, '-s', label='validation loss', markersize=12)
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
        
        # accuracy plot
        fig = plt.figure(num = 2, figsize = (12, 12), tight_layout = True)
        ax = fig.add_subplot(111)
        ax.plot(x_arr, self.mean_accuracy_train, '-o', label='train acc.', markersize=12)
        
        if self.mean_accuracy_valid is not None:
            ax.plot(x_arr, self.mean_accuracy_valid, '-s', label='validation acc.', markersize=12)
        else:
            pass
        
        plt.xlim((0, len(self.mean_loss_train) + 1))
        plt.ylim((0, 1))
        plt.xlabel('epoch')
        plt.ylabel('accuracy')
        plt.title('Model mean accuracy')
        plt.legend(loc = 'best')
        ax.grid(True)
        ax.label_outer()            
        ax.tick_params(which='both', direction='in',width=2)
        ax.tick_params(which='major', direction='in',length=7)
        ax.tick_params(which='minor', direction='in',length=4)
        ax.xaxis.set_ticks_position('both')
        ax.yaxis.set_ticks_position('both')
        plt.show()
        
    
    def test_model(self, test_dataset, model = None): #!!!
        """
        Test of the CNN model after the training.
        ------------------------------------------------------
        Par:
            test_dataset (torch.utils.data.DataLoader): test dataset
            model (torch): CNN model (if None, the used model is the one
                           in the train_model module, default = None)
        return:
             (float): 
        """
        
        # waits for all kernels in all streams on a CUDA device to complete 
        torch.cuda.synchronize()
        
        # to cpu (for the test)
        if model is not None:
            model = model
        else:
            model = self.model.cpu()
        
        # test CNN
        pred = model(test_dataset)
        is_correct = (torch.argmax(pred, dim=1) == test_dataset.targets).float().numpy()
        print(f'Test accuracy: {is_correct.mean():.4f}')
    
    
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
        
        filepath = "D:/Home/Universita'/Universita'/Magistrale/Master Thesis/Tesi_Codes/Prova_cnn/"
        
        checkpoint = {'model': self.model,
                      'state_dict': self.model.state_dict(),
                      'epochs': epochs,
                      'learning_rate': learning_rate,
                      'mean_loss_train': self.mean_loss_train,
                      'mean_accuracy_train': self.mean_accuracy_train,
                      'mean_loss_valid': self.mean_loss_valid,
                      'mean_accuracy_valid': self.mean_accuracy_valid,
                      'device': self.device,
                      'optimizer_state': self.optimizer.state_dict(),
                      'batch_size': batch_size}

        torch.save(checkpoint, filepath + filename)
    
    
    def load_model(self, filepath):
        """
        To load the CNN model (or a checkpoint to continue the CNN training).
        ------------------------------------------------------
        Par:
            filepath (str): path where to save the CNN model (.pt or .pth)
            
        return:
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
               'mean_accuracy_train': checkpoint['mean_accuracy_train'],
               'mean_loss_valid': checkpoint['mean_loss_valid'],
               'mean_accuracy_valid': checkpoint['mean_accuracy_valid'],
               'batch_size': checkpoint['batch_size']}
        
        return model, obs
    

# end