import torch
import torch.nn as nn
import torch.nn.functional as F

import torchdyn

class DeepTINet(nn.Module):
    '''
    Original DeepTI ConvNet in Maskey 2020.
    '''
    def __init__(self, image_channels, num_outputs):
        super(DeepTINet, self).__init__()
        self.conv1 = nn.Conv2d(1, 16, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=1)
        self.conv3 = nn.Conv2d(32, 32, kernel_size=4, stride=2, padding=1)
        self.flatten = nn.Flatten()
        self.output_dense = nn.Sequential(
            nn.LazyLinear(128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 1)
        )
    
    def forward(self, x):
        x = F.max_pool2d(F.relu(self.conv1(x)), kernel_size=3, stride=2)
        x = F.max_pool2d(F.relu(self.conv2(x)), kernel_size=3, stride=2)
        x = F.max_pool2d(F.relu(self.conv3(x)), kernel_size=3, stride=2)
        x = self.flatten(x)
        x = self.output_dense(x)
        return x

    