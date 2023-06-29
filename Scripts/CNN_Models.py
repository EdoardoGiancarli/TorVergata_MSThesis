# -*- coding: utf-8 -*-
"""
Created on Fri May 26 19:21:47 2023

@author: Edoardo Giancarli
"""

import torch.nn as nn

# loss and accuracy of the last element (so last epoch)

#### 1st model ################################################################

# mean loss (train) min: 0.688 (0.687 with dropout)
#
# mean accuracy (train) max: 0.54 (0.53 with dropout)
#
# mean loss (validation) min: 0.693 (0.686  with dropout)
#
# mean loss (validation) max: 0.6 (0.55 with dropout)

class GW_Deep1(nn.Module):
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
        [2] F. Fleuret, Deep Learning Course 14x050, University of Geneva,
            url: https://fleuret.org/dlc/
    """

    def __init__(self):                    
        super(GW_Deep1, self).__init__()
        
        # 1st layer (convolutional)
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=32, kernel_size=5, padding=2)
        self.ReLU1 = nn.ReLU()
        self.pool1 = nn.MaxPool2d(kernel_size=2)
        
        # 2nd layer (convolutional)
        self.conv2 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=5, padding=2)
        self.ReLU2 = nn.ReLU()
        self.pool2 = nn.MaxPool2d(kernel_size=2)
        
        # flatten operation
        self.flat = nn.Flatten()
        
        # 3rd layer (linear)
        self.linear3 = nn.Linear(in_features=65536, out_features=1024)  # in_features = channels x image_h/4 x image_w/4
        self.ReLU3 = nn.ReLU()
        self.drop3 = nn.Dropout(p=0.5)
        
        # 4th layer (linear, for output)
        self.linear4 = nn.Linear(in_features=1024, out_features=2)
    
        
    def forward(self, x):
        
        # 1st layer (convolutional)
        x = self.conv1(x)
        x = self.ReLU1(x)
        x = self.pool1(x)
        
        # 2nd layer (convolutional)
        x = self.conv2(x)
        x = self.ReLU2(x)
        x = self.pool2(x)
        
        # flat the output from the conv layers
        x = self.flat(x)
        
        # 3rd layer (linear)
        x = self.linear3(x)
        x = self.ReLU3(x)
        x = self.drop3(x)
        
        # 4th layer (linear, for output)
        x = self.linear4(x)
        
        return x


#### 2nd model ################################################################

# mean loss (train) min: 0.681
#
# mean accuracy (train) max: 0.57
#
# mean loss (validation) min: 0.680
#
# mean loss (validation) max: 0.675

class GW_Deep2(nn.Module):
    """
    CNN model for (binary) classification.
    ---------------------------------------------------------------------------
    Layers:
        1. Convolutional + ReLU activ. func. + Maxpooling
        2. Convolutional + ReLU activ. func. + Maxpooling + Dropout
        3. Convolutional + ReLU activ. func. + Maxpooling
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
        
        # 1st layer (convolutional)
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=16, kernel_size=5, padding=2)
        self.ReLU1 = nn.ReLU()
        self.pool1 = nn.MaxPool2d(kernel_size=2)
        
        # 2nd layer (convolutional)
        self.conv2 = nn.Conv2d(in_channels=16, out_channels=32, kernel_size=5, padding=2)
        self.ReLU2 = nn.ReLU()
        self.pool2 = nn.MaxPool2d(kernel_size=2)
        self.drop2 = nn.Dropout2d(p=0.5)
        
        # 3rd layer (convolutional)
        self.conv3 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=5, padding=2)
        self.ReLU3 = nn.ReLU()
        self.pool3 = nn.MaxPool2d(kernel_size=2)
        
        # flatten operation
        self.flat = nn.Flatten()
        
        # 4th layer (linear)
        self.linear4 = nn.Linear(in_features=16384, out_features=1024)  # in_features = channels x image_h/4 x image_w/4
        self.ReLU4 = nn.ReLU()
        self.drop4 = nn.Dropout(p=0.5)
        
        # 5th layer (linear, for output)
        self.linear5 = nn.Linear(in_features=1024, out_features=2)
    
        
    def forward(self, x):
        
        # 1st layer (convolutional)
        x = self.conv1(x)
        x = self.ReLU1(x)
        x = self.pool1(x)
        
        # 2nd layer (convolutional)
        x = self.conv2(x)
        x = self.ReLU2(x)
        x = self.pool2(x)
        x = self.drop2(x)
        
        # 3rd layer (convolutional)
        x = self.conv3(x)
        x = self.ReLU3(x)
        x = self.pool3(x)
        
        # flat the output from the conv layers
        x = self.flat(x)
        
        # 4th layer (linear)
        x = self.linear4(x)
        x = self.ReLU4(x)
        x = self.drop4(x)
        
        # 5th layer (linear, for output)
        x = self.linear5(x)
        
        return x


#### 3rd model ################################################################

# mean loss (train): 0.1 (0.157 batch_norm after ReLU)
#
# mean accuracy (train): 0.976 (0.95 batch_norm after ReLU)
#
# mean loss (validation): 3.75 (0.35 batch_norm after ReLU)
#
# mean loss (validation): 0.5 (0.8 batch_norm after ReLU)

class GW_Deep3(nn.Module):
    """
    CNN model for (binary) classification.
    ---------------------------------------------------------------------------
    Layers:
        1. Convolutional + BatchNorm + ReLU activ. func. + Maxpooling
        2. Convolutional + ReLU activ. func. + Maxpooling + Dropout
        3. Convolutional + ReLU activ. func. + Maxpooling
        4. Linear + ReLU + Dropout
        5. Linear
    
    Ref:
        [1] S. Raschka, Y. H. Liu, V. Mirjalili "Machine Learning with PyTorch
            and Scikit-Learn" (2022)
        [2] F. Fleuret, Deep Learning Course 14x050, University of Geneva,
            url: https://fleuret.org/dlc/
    """

    def __init__(self):                    
        super(GW_Deep3, self).__init__()
        
        # 1st layer (convolutional)
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=16, kernel_size=5, padding=2)
        self.bn1 = nn.BatchNorm2d(num_features=16)
        self.ReLU1 = nn.ReLU()
        self.pool1 = nn.MaxPool2d(kernel_size=2)
        
        # 2nd layer (convolutional)
        self.conv2 = nn.Conv2d(in_channels=16, out_channels=32, kernel_size=5, padding=2)
        self.ReLU2 = nn.ReLU()
        self.pool2 = nn.MaxPool2d(kernel_size=2)
        self.drop2 = nn.Dropout2d(p=0.5)
        
        # 3rd layer (convolutional)
        self.conv3 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=5, padding=2)
        self.ReLU3 = nn.ReLU()
        self.pool3 = nn.MaxPool2d(kernel_size=2)
        
        # flatten operation
        self.flat = nn.Flatten()
        
        # 4th layer (linear)
        self.linear4 = nn.Linear(in_features=16384, out_features=1024)  # in_features = channels x image_h/4 x image_w/4
        self.ReLU4 = nn.ReLU()
        self.drop4 = nn.Dropout(p=0.5)
        
        # 5th layer (linear, for output)
        self.linear5 = nn.Linear(in_features=1024, out_features=2)
    
        
    def forward(self, x):
        
        # 1st layer (convolutional)
        x = self.conv1(x)
        x = self.ReLU1(x)
        x = self.pool1(x)
        
        # 2nd layer (convolutional)
        x = self.conv2(x)
        x = self.ReLU2(x)
        x = self.pool2(x)
        x = self.drop2(x)
        
        # 3rd layer (convolutional)
        x = self.conv3(x)
        x = self.ReLU3(x)
        x = self.pool3(x)
        
        # flat the output from the conv layers
        x = self.flat(x)
        
        # 4th layer (linear)
        x = self.linear4(x)
        x = self.ReLU4(x)
        x = self.drop4(x)
        
        # 5th layer (linear, for output)
        x = self.linear5(x)
        
        return x


#### 4th model ################################################################

# maybe overfit

class GW_Deep4(nn.Module):
    """
    CNN model for (binary) classification.
    ---------------------------------------------------------------------------
    Layers:
        1. Convolutional + ReLU activ. func. + BatchNorm + Maxpooling
        2. Convolutional + ReLU activ. func. + BatchNorm + Maxpooling
        3. Convolutional + ReLU activ. func. + Maxpooling
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



# end