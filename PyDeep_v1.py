# -*- coding: utf-8 -*-
"""
Created on Sat May 13 20:27:28 2023

@author: Edoardo Giancarli
"""

#### ModuleMSThesis - CNN for spectrograms classification version 1 ###############################################

####   libraries   #####

import numpy as np                                # operations

import torch                                      # pytorch 
import torch.nn as nn

import matplotlib.pyplot as plt                   # plotting
from astroML.plotting import setup_text_plots
setup_text_plots(fontsize=26, usetex=True)

####    content    #####

# GW_Deep (class): CNN model for spectrograms (binary) classification
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
        1. Convolutional + ReLU activ. func. + Maxpooling
        2. Convolutional + ReLU activ. func. + Maxpooling
        3. Linear + ReLU + Dropout
        4. Linear
    
    Ref:
        [1] S. Raschka, Y. H. Liu, V. Mirjalili "Machine Learning with PyTorch
            and Scikit-Learn" (2022)
    """

    def __init__(self):                    #!!! values inside layers
        super(GW_Deep, self).__init__()
        
        # 1st layer (convolutional)
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=32, kernel_size=5, padding=2)
        self.pool1 = nn.MaxPool2d(kernel_size=2)
        
        # 2nd layer (convolutional)
        self.conv2 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=5, padding=2)
        self.pool2 = nn.MaxPool2d(kernel_size=2)
        
        # 3rd layer (linear)
        self.linear3 = nn.Linear(3136, 1024)
        self.dropout = nn.Dropout(p=0.5)
        
        # 4th layer (linear, for output)
        self.linear4 = nn.Linear(1024, 10)
    
        
    def forward(self, x):
        
        # 1st layer (convolutional)
        x = self.conv1(x)
        x = nn.ReLU(x)
        x = self.pool1(x)
        
        # 2nd layer (convolutional)
        x = self.conv2(x)
        x = nn.ReLU(x)
        x = self.pool2(x)
        
        # flat the output from the conv layers
        x = nn.Flatten(x)
        
        # 3rd layer (linear)
        x = self.linear3(x)
        x = nn.ReLU(x)
        x = self.dropout(x)
        
        # 4th layer (linear, for output)
        x = self.linear4(x)
        
        return x


#############################################################################################################################

#### class: CNN model investigation

class DeepTools:
    """
    This class contains the functions to train the CNN model, to plot the loss and the
    accuracy of the model, to test the CNN and to save/load the trained model.
    ---------------------------------------------------------------------------
    Attributes:
        

    Methods:
        train_model: it trains the CNN model defined in GW_Deep
        show_model: it shows the loss and the accuracy of the trained CNN model
        test_model: it tests the CNN model after the training
        save_model: it saves the CNN model (or if you want to save a checkpoint during training)
        load_model: it loads the saved CNN model (or the checkpoint to continue the CNN training)
    
    Ref:
        [1] S. Raschka, Y. H. Liu, V. Mirjalili "Machine Learning with PyTorch
            and Scikit-Learn" (2022)
    """

    def __init__(self, model):
        
        self.model = model
                
        self.loss_fn = nn.CrossEntropyLoss()
        self.optimizer = torch.optim.Adam()  # !!! does it work this way?
    
    
    def train_model(self, epochs, lr, train_dataset, valid_dataset = None):
        """
        Training of the CNN model defined in GW_Deep.
        ------------------------------------------------------
        Par:
            epochs (int): number of iterations for the model training
            lr (float): learning rate parameter for the model optimization
            train_dataset (torch.utils.data.DataLoader): training dataset
            valid_dataset (torch.utils.data.DataLoader): validation dataset (default = None)
            
        return:
            mean_loss_train (list): mean loss values for the model training
            mean_accuracy_train (list): mean accuracy values for the model training
            mean_loss_valid (list): mean loss values for the model validation (if valid_dataset is inserted)
            mean_accuracy_valid (list): mean accuracy values for the model validation (if valid_dataset is inserted)
        """
        
        # print loss and accuracy
        print_every = epochs/10
        
        # control device
        if torch.cuda.is_available():
            device = torch.device("cuda:0")
            print("Training on GPU...")
            self.model = self.model.to(device)
        
        else:
            print("No GPU available, training on CPU...\n")
            user_input = input("Continue training on CPU? (y/n): ")
            
            if user_input.lower() == "n":
                raise Exception("Training interrupted")
            else:
                device = torch.device("cpu")
        
        # define lists for loss and accuracy (both train and validation)
        mean_loss_train = [0]*epochs
        mean_accuracy_train = [0]*epochs
        mean_loss_valid = [0]*epochs
        mean_accuracy_valid = [0]*epochs
        
        # training loop
        for epoch in range(epochs):
            
            # model training
            self.model.train()
            
            for x_batch, y_batch in train_dataset:
                x_batch = x_batch.to(device) 
                y_batch = y_batch.to(device)
                
                pred = self.model(x_batch)                                        # model prediction
                loss = self.loss_fn(pred, y_batch)                                # model loss
                loss.backward()                                                   # backward propagation 
                self.optimizer(self.model.parameters(), lr=lr).step()             # model parameters optimization
                self.optimizer.zero_grad()                                        # put the optimizer grad to zero
                
                mean_loss_train[epoch] += loss.item()*y_batch.size(0)             # store single loss values 
                is_correct = (torch.argmax(pred, dim=1) == y_batch).float()
                mean_accuracy_train[epoch] += is_correct.sum()                    # store single accuracy values

            mean_loss_train[epoch] /= len(train_dataset.dataset)                  # compute the mean loss value for the epoch
            mean_accuracy_train[epoch] /= len(train_dataset.dataset)              # compute the mean accuracy value for the epoch  
            
            # model validation
            if valid_dataset is not None:
            
                self.model.eval()
                
                with torch.no_grad():
                    for x_batch, y_batch in valid_dataset:
                        x_batch = x_batch.to(device) 
                        y_batch = y_batch.to(device)
                        
                        valid_pred = self.model(x_batch)                                   # model prediction for validation
                        valid_loss = self.loss_fn(valid_pred, y_batch)                     # model loss for validation
                        mean_loss_valid[epoch] += valid_loss.item()*y_batch.size(0)        # store single validation loss values
                        is_correct = (torch.argmax(valid_pred, dim=1) == y_batch).float() 
                        mean_accuracy_valid[epoch] += is_correct.sum()                     # store single validation accuracy values
    
                mean_loss_valid[epoch] /= len(valid_dataset.dataset)                       # compute the validation mean loss value for the epoch
                mean_accuracy_valid[epoch] /= len(valid_dataset.dataset)                   # compute the validation mean accuracy value for the epoch
                
                if epoch % print_every == 0:
                    print(f"####### Epochs: {epoch + 1}/{epochs} #######",
                          f"Training Loss: {mean_loss_train[epoch]:.4f}, Training Accuracy: {mean_accuracy_train[epoch]:.4f}",
                          f"Validation Loss: {mean_loss_valid[epoch]:.4f}, Validation Accuracy: {mean_accuracy_valid[epoch]:.4f}")
                
            else:
                if epoch % print_every == 0:
                    print(f"####### Epochs: {epoch + 1}/{epochs} #######",
                          f"Training Loss: {mean_loss_train[epoch]:.4f}, Training Accuracy: {mean_accuracy_train[epoch]:.4f}")
        
        if valid_dataset is not None:
            return mean_loss_train, mean_accuracy_train, mean_loss_valid, mean_accuracy_valid
        else:
            return mean_loss_train, mean_accuracy_train
    
    
    def show_model(self, mean_loss_train, mean_accuracy_train,
                   mean_loss_valid = None, mean_accuracy_valid = None):
        """
        Plots of the trained CNN model loss and accuracy.
        ------------------------------------------------------
        Par:
            mean_loss_train (list): mean loss values for the model training
            mean_accuracy_train (list): mean accuracy values for the model training
            mean_loss_valid (list): mean loss values for the model validation (default = None)
            mean_accuracy_valid (list): mean accuracy values for the model validation (default = None)
        """
        
        # define ascissa
        x_arr = np.arange(len(mean_loss_train)) + 1
        
        # loss plot
        plt.figure(num = 1, figsize=(8, 8))
        plt.plot(x_arr, mean_loss_train, '-o', label='Train loss')
        
        if mean_loss_valid is not None:
            plt.plot(x_arr, mean_loss_valid, '-<', label='Validation loss')
        else:
            pass
        
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.title('Model mean loss')
        plt.legend(True)
        plt.show()
        
        # accuracy plot
        plt.figure(num = 2, figsize=(8, 8))
        plt.plot(x_arr, mean_accuracy_train, '-o', label='Train acc.')
        
        if mean_accuracy_valid is not None:
            plt.plot(x_arr, mean_accuracy_valid, '-<', label='Validation acc.')
        else:
            pass
        
        plt.xlabel('Epoch')
        plt.ylabel('Accuracy')
        plt.title('Model mean accuracy')
        plt.legend(True)
        plt.show()
        
    
    def test_model(self):
        """
        Test of the CNN model after the training.
        ------------------------------------------------------
        Par:
            
            
        return:
            
        """
        
        torch.cuda.synchronize()
        
        return 1
    
    
    def save_model(self):
        """
        To save the CNN model (or a checkpoint during training).
        ------------------------------------------------------
        Par:
            
            
        return:
            
        """
        
        return 1
    
    
    def load_model(self):
        """
        To load the CNN model (or a checkpoint to continue the CNN training).
        ------------------------------------------------------
        Par:
            
            
        return:
            
        """
        
        return 1
    

# end