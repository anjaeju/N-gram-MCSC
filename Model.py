import numpy as np

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F


class Net(nn.Module):
    def __init__(self, image_shape, output):
        super(Net, self).__init__()

        self.conv = nn.Sequential(nn.Conv2d(image_shape[0], 32, kernel_size=2),
                                 
                                 nn.BatchNorm2d(32),
                                 nn.Conv2d(32, 32, kernel_size=2),
                                 nn.ReLU(),
                                 
                                 nn.BatchNorm2d(32),
                                 nn.Conv2d(32, 32, kernel_size=2),
                                 nn.ReLU())
                                 
        conv_out_size = self._get_conv_out(image_shape)
        
        self.fc = nn.Sequential(nn.Linear(conv_out_size, 512),
                               nn.ReLU(),
                               nn.Linear(512, 256),
                               nn.ReLU(),
                               nn.Linear(256, output))
    
    def _get_conv_out(self, shape):
        o = self.conv(torch.zeros(1, *shape))
        return int(np.prod(o.size()))
    
    def forward(self, x):
        conv_out = self.conv(x).view(x.size()[0], -1)
        return self.fc(conv_out)