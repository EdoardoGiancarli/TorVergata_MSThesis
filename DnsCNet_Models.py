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
        1. Convolution + PReLU activ. func.
        2. Convolution + PReLU activ. func.
        3. Convolution + PReLU activ. func.
        4. Convolution + PReLU activ. func.
        5. Convolution + PReLU activ. func.
    
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


#### model 1.1 ############################
class DnsCNet1_1(nn.Module):
    """
    CNN model for spectrograms denoising.
    ---------------------------------------------------------------------------
    Layers:
        1. Convolution (with xavier Gaussian kernels) + PReLU activ. func.
        2. Convolution + PReLU activ. func.
        3. Convolution + PReLU activ. func.
        4. Convolution + PReLU activ. func.
        5. Convolution + PReLU activ. func.
    
    Ref:
        [1] Andrew Reader, url: https://www.youtube.com/watch?v=Wq8mh3Y0JjA&t=1759s
        [2] X. Glorot, Y. Bengio, "Understanding the difficulty of training deep
            feedforward neural networks" (2010)
    """
    
    def __init__(self):
        super(DnsCNet1_1, self).__init__()
        
        conv1 = nn.Conv2d(in_channels=1, out_channels=8, kernel_size=7, padding=3)
        nn.init.xavier_normal_(conv1.weight)
        
        self.layers = nn.Sequential(
            conv1, nn.PReLU(),
            nn.Conv2d(in_channels=8, out_channels=8, kernel_size=7, padding=3), nn.PReLU(),
            nn.Conv2d(in_channels=8, out_channels=8, kernel_size=7, padding=3), nn.PReLU(),
            nn.Conv2d(in_channels=8, out_channels=8, kernel_size=7, padding=3), nn.PReLU(),
            nn.Conv2d(in_channels=8, out_channels=1, kernel_size=7, padding=3), nn.PReLU())
    
    
    def forward(self, x):
        x = self.layers(x)
        return x


#### model 1.2 ############################
class DnsCNet1_2(nn.Module):
    """
    CNN model for spectrograms denoising.
    ---------------------------------------------------------------------------
    Layers:
        1. Convolution + PReLU activ. func.
        2. Convolution + PReLU activ. func.
        3. Convolution + PReLU activ. func.
        4. Convolution + PReLU activ. func.
        5. Convolution + PReLU activ. func.
    
    Ref:
        [1] Andrew Reader, url: https://www.youtube.com/watch?v=Wq8mh3Y0JjA&t=1759s
    """
    
    def __init__(self):
        super(DnsCNet1_2, self).__init__()
        
        self.layers = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=8, kernel_size=7, padding=3), nn.PReLU(),
            nn.Conv2d(in_channels=8, out_channels=16, kernel_size=7, padding=3), nn.PReLU(),
            nn.Conv2d(in_channels=16, out_channels=32, kernel_size=7, padding=3), nn.PReLU(),
            nn.Conv2d(in_channels=32, out_channels=8, kernel_size=7, padding=3), nn.PReLU(),
            nn.Conv2d(in_channels=8, out_channels=1, kernel_size=7, padding=3), nn.PReLU())
    
    
    def forward(self, x):
        x = self.layers(x)
        return x
    

#### model 1.3 ############################
class DnsCNet1_3(nn.Module):
    """
    CNN model for spectrograms denoising.
    ---------------------------------------------------------------------------
    Layers:
        1. Convolution (with xavier Gaussian kernels) + PReLU activ. func.
        2. Convolution + PReLU activ. func.
        3. Convolution + PReLU activ. func.
        4. Convolution + PReLU activ. func.
        5. Convolution + PReLU activ. func.
    
    Ref:
        [1] Andrew Reader, url: https://www.youtube.com/watch?v=Wq8mh3Y0JjA&t=1759s
        [2] X. Glorot, Y. Bengio, "Understanding the difficulty of training deep
            feedforward neural networks" (2010)
    """
    
    def __init__(self):
        super(DnsCNet1_3, self).__init__()
        
        conv1 = nn.Conv2d(in_channels=1, out_channels=8, kernel_size=7, padding=3)
        nn.init.xavier_normal_(conv1.weight)
        
        self.layers = nn.Sequential(
            conv1, nn.PReLU(),
            nn.Conv2d(in_channels=8, out_channels=16, kernel_size=7, padding=3), nn.PReLU(),
            nn.Conv2d(in_channels=16, out_channels=32, kernel_size=7, padding=3), nn.PReLU(),
            nn.Conv2d(in_channels=32, out_channels=8, kernel_size=7, padding=3), nn.PReLU(),
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
        1. Convolution + PReLU activ. func. + Batch Norm
        2. Convolution + PReLU activ. func.
        3. Convolution + PReLU activ. func. + Batch Norm
        4. Convolution + PReLU activ. func.
        5. Convolution + PReLU activ. func.
    
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


#### model 2.1 ############################
class DnsCNet2_1(nn.Module):
    """
    CNN model for spectrograms denoising.
    ---------------------------------------------------------------------------
    Layers:
        1. Convolution (with xavier Gaussian kernels) + PReLU activ. func. + Batch Norm
        2. Convolution + PReLU activ. func.
        3. Convolution + PReLU activ. func. + Batch Norm
        4. Convolution + PReLU activ. func.
        5. Convolution + PReLU activ. func.
    
    Ref:
        [1] Andrew Reader, url: https://www.youtube.com/watch?v=Wq8mh3Y0JjA&t=1759s
        [2] X. Glorot, Y. Bengio, "Understanding the difficulty of training deep
            feedforward neural networks" (2010)
    """
    
    def __init__(self):
        super(DnsCNet2_1, self).__init__()
        
        conv1 = nn.Conv2d(in_channels=1, out_channels=8, kernel_size=7, padding=3)
        nn.init.xavier_normal_(conv1.weight)
        
        self.layers = nn.Sequential(
            conv1, nn.PReLU(), nn.BatchNorm2d(num_features=8),
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
        1. Convolution + PReLU activ. func. + Batch Norm
        2. Convolution + PReLU activ. func.
        3. Convolution + PReLU activ. func. + Dropout
        4. Convolution + PReLU activ. func. + Dropout
        5. Convolution + PReLU activ. func.
    
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
    

#### model 3.1 ############################
class DnsCNet3_1(nn.Module):
    """
    CNN model for spectrograms denoising.
    ---------------------------------------------------------------------------
    Layers:
        1. Convolution (with xavier Gaussian kernels) + PReLU activ. func. + Batch Norm
        2. Convolution + PReLU activ. func.
        3. Convolution + PReLU activ. func. + Dropout
        4. Convolution + PReLU activ. func. + Dropout
        5. Convolution + PReLU activ. func.
    
    Ref:
        [1] Andrew Reader, url: https://www.youtube.com/watch?v=Wq8mh3Y0JjA&t=1759s
        [2] X. Glorot, Y. Bengio, "Understanding the difficulty of training deep
            feedforward neural networks" (2010)
    """
    
    def __init__(self):
        super(DnsCNet3_1, self).__init__()
        
        conv1 = nn.Conv2d(in_channels=1, out_channels=8, kernel_size=7, padding=3)
        nn.init.xavier_normal_(conv1.weight)
        
        self.layers = nn.Sequential(
            conv1, nn.PReLU(), nn.BatchNorm2d(num_features=8),
            nn.Conv2d(in_channels=8, out_channels=8, kernel_size=7, padding=3), nn.PReLU(),
            nn.Conv2d(in_channels=8, out_channels=8, kernel_size=7, padding=3), nn.PReLU(), nn.Dropout2d(p=0.5),
            nn.Conv2d(in_channels=8, out_channels=8, kernel_size=7, padding=3), nn.PReLU(), nn.Dropout2d(p=0.5),
            nn.Conv2d(in_channels=8, out_channels=1, kernel_size=7, padding=3), nn.PReLU())
    
    
    def forward(self, x):
        x = self.layers(x)
        return x


#### model 3.2 ############################
class DnsCNet3_2(nn.Module):
    """
    CNN model for spectrograms denoising.
    ---------------------------------------------------------------------------
    Layers:
        1. Convolution + PReLU activ. func. + Batch Norm
        2. Convolution + PReLU activ. func.
        3. Convolution + PReLU activ. func. + Dropout
        4. Convolution + PReLU activ. func. + Dropout
        5. Convolution + PReLU activ. func.
    
    Ref:
        [1] Andrew Reader, url: https://www.youtube.com/watch?v=Wq8mh3Y0JjA&t=1759s
    """
    
    def __init__(self):
        super(DnsCNet3_2, self).__init__()
        
        self.layers = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=8, kernel_size=7, padding=3), nn.PReLU(), nn.BatchNorm2d(num_features=8),
            nn.Conv2d(in_channels=8, out_channels=16, kernel_size=7, padding=3), nn.PReLU(),
            nn.Conv2d(in_channels=16, out_channels=32, kernel_size=7, padding=3), nn.PReLU(), nn.Dropout2d(p=0.5),
            nn.Conv2d(in_channels=32, out_channels=8, kernel_size=7, padding=3), nn.PReLU(), nn.Dropout2d(p=0.5),
            nn.Conv2d(in_channels=8, out_channels=1, kernel_size=7, padding=3), nn.PReLU())
    
    
    def forward(self, x):
        x = self.layers(x)
        return x
    

#### model 3.3 ############################
class DnsCNet3_3(nn.Module):
    """
    CNN model for spectrograms denoising.
    ---------------------------------------------------------------------------
    Layers:
        1. Convolution (with xavier Gaussian kernels) + PReLU activ. func. + Batch Norm
        2. Convolution + PReLU activ. func.
        3. Convolution + PReLU activ. func. + Dropout
        4. Convolution + PReLU activ. func. + Dropout
        5. Convolution + PReLU activ. func.
    
    Ref:
        [1] Andrew Reader, url: https://www.youtube.com/watch?v=Wq8mh3Y0JjA&t=1759s
        [2] X. Glorot, Y. Bengio, "Understanding the difficulty of training deep
            feedforward neural networks" (2010)
    """
    
    def __init__(self):
        super(DnsCNet3_3, self).__init__()
        
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


# end