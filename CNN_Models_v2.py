# -*- coding: utf-8 -*-
"""
Created on Sun Jun 11 02:42:13 2023

@author: Edoardo Giancarli
"""

import torch.nn as nn

# loss and accuracy of the last element (so last epoch)
#
# for the linear layers after the nn.Flatten() operation: in_features = channels x image_h/4 x image_w/4

#### 1st model ################################################################
class GW_Deep1(nn.Module):
    """
    CNN model for (binary) classification.
    ---------------------------------------------------------------------------
    Layers:
        1. Convolution + ReLU activ. func. + Maxpooling
        2. Convolution + ReLU activ. func. + Maxpooling
        3. Linear + ReLU + Dropout
        4. Linear
    
    Ref:
        [1] S. Raschka, Y. H. Liu, V. Mirjalili "Machine Learning with PyTorch
            and Scikit-Learn" (2022)
        [2] F. Fleuret, Deep Learning Course 14x050, University of Geneva,
            url: https://fleuret.org/dlc/
    """
    
    def __init__(self):
        super(GW_Deep1, self).__init__()
        
        self.layers = nn.Sequential(
            # 1st layer (convolutional)
            nn.Conv2d(in_channels=1, out_channels=32, kernel_size=5, padding=2), nn.ReLU(), nn.MaxPool2d(kernel_size=2),
            
            # 2nd layer (convolutional)
            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=5, padding=2), nn.ReLU(), nn.MaxPool2d(kernel_size=2),
            
            # flatten operation
            nn.Flatten(),
            
            # 3rd layer (linear)
            nn.Linear(in_features=65536, out_features=1024), nn.ReLU(), nn.Dropout(p=0.5),
            
            # 4th layer (linear, for output)
            nn.Linear(in_features=1024, out_features=2))
    
    
    def forward(self, x):
        x = self.layers(x)
        return x


#### 2nd model ################################################################
class GW_Deep2(nn.Module):
    """
    CNN model for (binary) classification.
    ---------------------------------------------------------------------------
    Layers:
        1. Convolution + ReLU activ. func. + Maxpooling
        2. Convolution + ReLU activ. func. + Maxpooling + Dropout
        3. Convolution + ReLU activ. func. + Maxpooling
        4. Linear + ReLU + Dropout
        5. Linear
    
    Ref:
        [1] S. Raschka, Y. H. Liu, V. Mirjalili "Machine Learning with PyTorch
            and Scikit-Learn" (2022)
        [2] F. Fleuret, Deep Learning Course 14x050, University of Geneva,
            url: https://fleuret.org/dlc/
    """
    
    def __init__(self):
        super(GW_Deep2, self).__init__()
        
        self.layers = nn.Sequential(
            # 1st layer (convolutional)
            nn.Conv2d(in_channels=1, out_channels=16, kernel_size=5, padding=2), nn.ReLU(), nn.MaxPool2d(kernel_size=2),
            
            # 2nd layer (convolutional)
            nn.Conv2d(in_channels=16, out_channels=32, kernel_size=5, padding=2), nn.ReLU(), nn.MaxPool2d(kernel_size=2), nn.Dropout2d(p=0.5),
            
            # 3rd layer (convolutional)
            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=5, padding=2), nn.ReLU(), nn.MaxPool2d(kernel_size=2),
            
            # flatten operation
            nn.Flatten(),
            
            # 4th layer (linear)
            nn.Linear(in_features=16384, out_features=1024), nn.ReLU(), nn.Dropout(p=0.5),
            
            # 5th layer (linear, for output)
            nn.Linear(in_features=1024, out_features=2))
    
    
    def forward(self, x):
        x = self.layers(x)
        return x


#### 3rd model ################################################################
class GW_Deep3(nn.Module):
    """
    CNN model for (binary) classification.
    ---------------------------------------------------------------------------
    Layers:
        1. Convolution + BatchNorm + ReLU activ. func. + Maxpooling
        2. Convolution + ReLU activ. func. + Maxpooling + Dropout
        3. Convolution + ReLU activ. func. + Maxpooling
        4. Linear + ReLU + Dropout
        5. Linear
    
    Ref:
        [1] S. Raschka, Y. H. Liu, V. Mirjalili "Machine Learning with PyTorch
            and Scikit-Learn" (2022)
        [2] F. Fleuret, Deep Learning Course 14x050, University of Geneva,
            url: https://fleuret.org/dlc/
    """
    
    def __init__(self):
        super(GW_Deep4, self).__init__()
        
        self.layers = nn.Sequential(
            # 1st layer (convolutional)
            nn.Conv2d(in_channels=1, out_channels=16, kernel_size=5, padding=2), nn.BatchNorm2d(num_features=16), nn.ReLU(), nn.MaxPool2d(kernel_size=2),
            
            # 2nd layer (convolutional)
            nn.Conv2d(in_channels=16, out_channels=32, kernel_size=5, padding=2), nn.ReLU(), nn.MaxPool2d(kernel_size=2), nn.Dropout2d(p=0.5),
            
            # 3rd layer (convolutional)
            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=5, padding=2), nn.ReLU(), nn.MaxPool2d(kernel_size=2),
            
            # flatten operation
            nn.Flatten(),
            
            # 4th layer (linear)
            nn.Linear(in_features=16384, out_features=1024), nn.ReLU(), nn.Dropout(p=0.5),
            
            # 5th layer (linear, for output)
            nn.Linear(in_features=1024, out_features=2))
    
    
    def forward(self, x):
        x = self.layers(x)
        return x


#### 4th model ################################################################
class GW_Deep4(nn.Module):
    """
    CNN model for (binary) classification.
    ---------------------------------------------------------------------------
    Layers:
        1. Convolution + ReLU activ. func. + BatchNorm + Maxpooling
        2. Convolution + ReLU activ. func. + BatchNorm + Maxpooling
        3. Convolution + ReLU activ. func. + Maxpooling
        4. Linear + ReLU + BatchNorm
        5. Linear
    
    Ref:
        [1] S. Raschka, Y. H. Liu, V. Mirjalili "Machine Learning with PyTorch
            and Scikit-Learn" (2022)
        [2] F. Fleuret, Deep Learning Course 14x050, University of Geneva,
            url: https://fleuret.org/dlc/
    """
    
    def __init__(self):
        super(GW_Deep4, self).__init__()
        
        self.layers = nn.Sequential(
            # 1st layer (convolutional)
            nn.Conv2d(in_channels=1, out_channels=16, kernel_size=5, padding=2), nn.ReLU(), nn.BatchNorm2d(num_features=16), nn.MaxPool2d(kernel_size=2),
            
            # 2nd layer (convolutional)
            nn.Conv2d(in_channels=16, out_channels=32, kernel_size=5, padding=2), nn.ReLU(), nn.BatchNorm2d(num_features=32), nn.MaxPool2d(kernel_size=2),
            
            # 3rd layer (convolutional)
            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=5, padding=2), nn.ReLU(), nn.MaxPool2d(kernel_size=2),
            
            # flatten operation
            nn.Flatten(),
            
            # 4th layer (linear)
            nn.Linear(in_features=16384, out_features=1024), nn.ReLU(), nn.BatchNorm1d(num_features=1024),
            
            # 5th layer (linear, for output)
            nn.Linear(in_features=1024, out_features=2))
    
    
    def forward(self, x):
        x = self.layers(x)
        return x


#### 5th model ################################################################
class GW_Deep5(nn.Module):
    """
    CNN model for (binary) classification.
    ---------------------------------------------------------------------------
    Layers:
        1. Convolution + PReLU activ. func. + Maxpooling + BatchNorm
        2. Convolution + PReLU activ. func. + Maxpooling + Dropout
        3. Linear + PReLU + Dropout
        4. Linear
    
    Ref:
        [1] S. Raschka, Y. H. Liu, V. Mirjalili "Machine Learning with PyTorch
            and Scikit-Learn" (2022)
        [2] F. Fleuret, Deep Learning Course 14x050, University of Geneva,
            url: https://fleuret.org/dlc/
    """
    
    def __init__(self):
        super(GW_Deep5, self).__init__()
        
        self.layers = nn.Sequential(
            # 1st layer (convolutional)
            nn.Conv2d(in_channels=1, out_channels=16, kernel_size=5, padding=2), nn.PReLU(), nn.MaxPool2d(kernel_size=2), nn.BatchNorm2d(num_features=16),
            
            # 2nd layer (convolutional)
            nn.Conv2d(in_channels=16, out_channels=32, kernel_size=5, padding=2), nn.PReLU(), nn.MaxPool2d(kernel_size=2), nn.Dropout2d(p=0.5),
            
            # flatten operation
            nn.Flatten(),
            
            # 3th layer (linear)
            nn.Linear(in_features=32768, out_features=1024), nn.PReLU(), nn.Dropout(p=0.5),
            
            # 4th layer (linear, for output)
            nn.Linear(in_features=1024, out_features=2))
    
    
    def forward(self, x):
        x = self.layers(x)
        return x


#### 6th model ################################################################
class GW_Deep6(nn.Module):
    """
    CNN model for (binary) classification.
    ---------------------------------------------------------------------------
    Layers:
        1. Convolution + PReLU activ. func. + Dropout
        2. Convolution + PReLU activ. func. + Maxpooling
        3. Linear + PReLU + Dropout
        4. Linear
    
    Ref:
        [1] S. Raschka, Y. H. Liu, V. Mirjalili "Machine Learning with PyTorch
            and Scikit-Learn" (2022)
        [2] F. Fleuret, Deep Learning Course 14x050, University of Geneva,
            url: https://fleuret.org/dlc/
    """
    
    def __init__(self):
        super(GW_Deep6, self).__init__()
        
        self.layers = nn.Sequential(
            # 1st layer (convolutional)
            nn.Conv2d(in_channels=1, out_channels=8, kernel_size=5, padding=2), nn.PReLU(), nn.Dropout2d(p=0.5),
            
            # 2nd layer (convolutional)
            nn.Conv2d(in_channels=8, out_channels=16, kernel_size=5, padding=2), nn.PReLU(), nn.MaxPool2d(kernel_size=2),
            
            # flatten operation
            nn.Flatten(),
            
            # 3th layer (linear)
            nn.Linear(in_features=65536, out_features=1024), nn.PReLU(), nn.Dropout(p=0.5),
            
            # 4th layer (linear, for output)
            nn.Linear(in_features=1024, out_features=2))
    
    
    def forward(self, x):
        x = self.layers(x)
        return x


#### 7th model ################################################################
class GW_Deep7(nn.Module):
    """
    CNN model for (binary) classification.
    ---------------------------------------------------------------------------
    Layers:
        1. Convolution + PReLU activ. func. + Batch Norm
        2. Convolution + PReLU activ. func. + Maxpooling
        3. Linear + PReLU + Dropout
        4. Linear
    
    Ref:
        [1] S. Raschka, Y. H. Liu, V. Mirjalili "Machine Learning with PyTorch
            and Scikit-Learn" (2022)
        [2] F. Fleuret, Deep Learning Course 14x050, University of Geneva,
            url: https://fleuret.org/dlc/
    """
    
    def __init__(self):
        super(GW_Deep, self).__init__()
        
        self.layers = nn.Sequential(
            # 1st layer (convolutional)
            nn.Conv2d(in_channels=1, out_channels=8, kernel_size=5, padding=2), nn.PReLU(), nn.BatchNorm2d(num_features=8),
            
            # 2nd layer (convolutional)
            nn.Conv2d(in_channels=8, out_channels=16, kernel_size=5, padding=2), nn.PReLU(), nn.MaxPool2d(kernel_size=2),
            
            # flatten operation
            nn.Flatten(),
            
            # 3th layer (linear)
            nn.Linear(in_features=65536, out_features=1024), nn.PReLU(), nn.Dropout(p=0.5),
            
            # 4th layer (linear, for output)
            nn.Linear(in_features=1024, out_features=2))
    
    
    def forward(self, x):
        x = self.layers(x)
        return x


#### model 7.1 #########################
class GW_Deep7_1(nn.Module):
    """
    CNN model for (binary) classification.
    ---------------------------------------------------------------------------
    Layers:
        1. Convolution + PReLU activ. func. + Batch Norm
        2. Convolution + PReLU activ. func. + Maxpooling
        3. Linear + PReLU + Dropout
        4. Linear
    
    Ref:
        [1] S. Raschka, Y. H. Liu, V. Mirjalili "Machine Learning with PyTorch
            and Scikit-Learn" (2022)
        [2] F. Fleuret, Deep Learning Course 14x050, University of Geneva,
            url: https://fleuret.org/dlc/
    """
    
    def __init__(self):
        super(GW_Deep7_1, self).__init__()
        
        self.layers = nn.Sequential(
            # 1st layer (convolutional)
            nn.Conv2d(in_channels=1, out_channels=8, kernel_size=5, padding=2), nn.PReLU(), nn.BatchNorm2d(num_features=8),
            
            # 2nd layer (convolutional)
            nn.Conv2d(in_channels=8, out_channels=1, kernel_size=5, padding=2), nn.PReLU(), nn.MaxPool2d(kernel_size=2),
            
            # flatten operation
            nn.Flatten(),
            
            # 3th layer (linear)
            nn.Linear(in_features=4096, out_features=1024), nn.PReLU(), nn.Dropout(p=0.5),
            
            # 4th layer (linear, for output)
            nn.Linear(in_features=1024, out_features=2))
    
    
    def forward(self, x):
        x = self.layers(x)
        return x


# end