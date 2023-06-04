# -*- coding: utf-8 -*-
"""
Created on Sat Jun  3 16:08:40 2023

@author: Edoardo Giancarli
"""

import torch.nn as nn

# plot of model mean loss and loss change rate in folder: "D:/Home/Universita'/Universita'/Magistrale/Master Thesis/Tesi_Codes/Prova_Denoiser"

#### 1st model ################################################################
class DnsCNet1(nn.Module):
    """
    CNN model for spectrograms denoising.
    ---------------------------------------------------------------------------
    Layers:
        1. Convolutional + PReLU activ. func.
        2. Convolutional + PReLU activ. func.
        3. Convolutional + PReLU activ. func.
        4. Convolutional + PReLU activ. func.
        5. Convolutional + PReLU activ. func.
    
    Ref:
        [1] Andrew Reader, url: https://www.youtube.com/watch?v=Wq8mh3Y0JjA&t=1759s
    """
    
    def __init__(self):
        super(DnsCNet1, self).__init__()
        
        self.layers = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=8, kernel_size=7, padding=3), nn.PReLU(),
            nn.Conv2d(in_channels=8, out_channels=8, kernel_size=7, padding=3), nn.PReLU(),
            nn.Conv2d(in_channels=8, out_channels=8, kernel_size=7, padding=3), nn.PReLU(),
            nn.Conv2d(in_channels=8, out_channels=8, kernel_size=7, padding=3), nn.PReLU(),
            nn.Conv2d(in_channels=8, out_channels=1, kernel_size=7, padding=3), nn.PReLU())
    
    
    def forward(self, x):
        x = self.layers(x)
        return x


#### 2nd model ################################################################
class DnsCNet2(nn.Module):
    """
    CNN model for spectrograms denoising.
    ---------------------------------------------------------------------------
    Layers:
        1. Convolutional + PReLU activ. func. + Batch Norm
        2. Convolutional + PReLU activ. func.
        3. Convolutional + PReLU activ. func. + Batch Norm
        4. Convolutional + PReLU activ. func.
        5. Convolutional + PReLU activ. func.
    
    Ref:
        [1] Andrew Reader, url: https://www.youtube.com/watch?v=Wq8mh3Y0JjA&t=1759s
    """
    
    def __init__(self):
        super(DnsCNet2, self).__init__()
        
        self.layers = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=8, kernel_size=7, padding=3), nn.PReLU(), nn.BatchNorm2d(num_features=8),
            nn.Conv2d(in_channels=8, out_channels=8, kernel_size=7, padding=3), nn.PReLU(),
            nn.Conv2d(in_channels=8, out_channels=8, kernel_size=7, padding=3), nn.PReLU(), nn.BatchNorm2d(num_features=8),
            nn.Conv2d(in_channels=8, out_channels=8, kernel_size=7, padding=3), nn.PReLU(),
            nn.Conv2d(in_channels=8, out_channels=1, kernel_size=7, padding=3), nn.PReLU())
    
    
    def forward(self, x):
        x = self.layers(x)
        return x


#### 3rd model ################################################################
class DnsCNet3(nn.Module):
    """
    CNN model for spectrograms denoising.
    ---------------------------------------------------------------------------
    Layers:
        1. Convolutional + PReLU activ. func. + Batch Norm
        2. Convolutional + PReLU activ. func.
        3. Convolutional + PReLU activ. func. + Dropout
        4. Convolutional + PReLU activ. func. + Dropout
        5. Convolutional + PReLU activ. func.
    
    Ref:
        [1] Andrew Reader, url: https://www.youtube.com/watch?v=Wq8mh3Y0JjA&t=1759s
    """
    
    def __init__(self):
        super(DnsCNet3, self).__init__()
        
        self.layers = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=8, kernel_size=7, padding=3), nn.PReLU(), nn.BatchNorm2d(num_features=8),
            nn.Conv2d(in_channels=8, out_channels=8, kernel_size=7, padding=3), nn.PReLU(),
            nn.Conv2d(in_channels=8, out_channels=8, kernel_size=7, padding=3), nn.PReLU(), nn.Dropout2d(p=0.5),
            nn.Conv2d(in_channels=8, out_channels=8, kernel_size=7, padding=3), nn.PReLU(), nn.Dropout2d(p=0.5),
            nn.Conv2d(in_channels=8, out_channels=1, kernel_size=7, padding=3), nn.PReLU())
    
    
    def forward(self, x):
        x = self.layers(x)
        return x


#### 4th model ################################################################
class DnsCNet4(nn.Module):
    """
    CNN model for spectrograms denoising.
    ---------------------------------------------------------------------------
    Layers:
        1. Convolutional + PReLU activ. func.
        2. Convolutional + PReLU activ. func.
        3. Convolutional + PReLU activ. func.
        4. Convolutional + PReLU activ. func.
        5. Convolutional + PReLU activ. func.
    
    Ref:
        [1] Andrew Reader, url: https://www.youtube.com/watch?v=Wq8mh3Y0JjA&t=1759s
    """
    
    def __init__(self):
        super(DnsCNet4, self).__init__()
        
        self.layers = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=8, kernel_size=7, padding=3), nn.PReLU(),
            nn.Conv2d(in_channels=8, out_channels=16, kernel_size=7, padding=3), nn.PReLU(),
            nn.Conv2d(in_channels=16, out_channels=32, kernel_size=7, padding=3), nn.PReLU(),
            nn.Conv2d(in_channels=32, out_channels=8, kernel_size=7, padding=3), nn.PReLU(),
            nn.Conv2d(in_channels=8, out_channels=1, kernel_size=7, padding=3), nn.PReLU())
    
    
    def forward(self, x):
        x = self.layers(x)
        return x


#### 5th model ################################################################
class DnsCNet5(nn.Module):
    """
    CNN model for spectrograms denoising.
    ---------------------------------------------------------------------------
    Layers:
        1. Convolutional + PReLU activ. func. + Batch Norm
        2. Convolutional + PReLU activ. func.
        3. Convolutional + PReLU activ. func. + Dropout
        4. Convolutional + PReLU activ. func. + Dropout
        5. Convolutional + PReLU activ. func.
    
    Ref:
        [1] Andrew Reader, url: https://www.youtube.com/watch?v=Wq8mh3Y0JjA&t=1759s
    """
    
    def __init__(self):
        super(DnsCNet5, self).__init__()
        
        self.layers = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=8, kernel_size=7, padding=3), nn.PReLU(), nn.BatchNorm2d(num_features=8),
            nn.Conv2d(in_channels=8, out_channels=16, kernel_size=7, padding=3), nn.PReLU(),
            nn.Conv2d(in_channels=16, out_channels=32, kernel_size=7, padding=3), nn.PReLU(), nn.Dropout2d(p=0.5),
            nn.Conv2d(in_channels=32, out_channels=8, kernel_size=7, padding=3), nn.PReLU(), nn.Dropout2d(p=0.5),
            nn.Conv2d(in_channels=8, out_channels=1, kernel_size=7, padding=3), nn.PReLU())
    
    
    def forward(self, x):
        x = self.layers(x)
        return x










