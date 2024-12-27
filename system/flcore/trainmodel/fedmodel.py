import os 
import numpy as np 
import torch 
import torch.nn as nn 
import torch.nn.functional as F
import torch.optim as optim



class HybridModel(nn.Module):
    def __init__(self, in_channels:int = 9, dim_hidden=128, num_classes=6, conv_kernel_size = (1, 9), pool_kernel_size =2 ):
        self.in_channels = in_channels
        self.dim_hidden = dim_hidden
        self.num_classes = num_classes
        self.conv_kernel_size = conv_kernel_size
        self.pool_kernel_size = pool_kernel_size

        self.conv_module = nn.Sequential(
            nn.Conv2d(in_channels, 32, kernel_size=conv_kernel_size),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(pool_kernel_size)
        )
        
        self.lstm = nn.LSTM(32, dim_hidden, 2, batch_first=True)
        