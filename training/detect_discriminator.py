import copy
import math

import numpy as np
import torch
from torch import nn
import torch.nn.functional as F



class SatNet(torch.nn.Module):
	
    def __init__(self,
            c_dim,  # Conditioning label (C) dimensionality.
            img_resolution,  # Input resolution.
            img_channels, 
            device='cuda',
            block_kwargs={},  # Arguments for DiscriminatorBlock.
            mapping_kwargs={},  # Arguments for MappingNetwork.
            epilogue_kwargs={}):  # Arguments for DiscriminatorEpilogue.
        super(SatNet, self).__init__()
        # 3 input image channel, 6 output channels, 
        # 5x5 square convolution kernel
        self.conv1 = nn.Conv2d(1, 6, 5).to(device)
        # Max pooling over a (2, 2) window
        self.pool = nn.MaxPool2d(2, 2).to(device)
        self.conv2 = nn.Conv2d(6, 16, 5).to(device)
        self.fc1 = nn.Linear(16 * 253 * 253, 120).to(device)# 5x5 from image dimension
        self.fc2 = nn.Linear(120, 84).to(device)
        self.fc3 = nn.Linear(84, 2).to(device)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = torch.reshape(x, (-1,))
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x

