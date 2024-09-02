from typing import Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F


class Network(nn.Module):
    def __init__(self, image_dims: Tuple, output_dims: int):
        super(Network, self).__init__()
        self.image_dims = image_dims
        self.linear_size = 16 *(image_dims[1] // 4) * (image_dims[2] // 4)

        self.conv1 = nn.Conv2d(3, 6, 5, padding=2)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5, padding=2)
        self.fc1 = nn.Linear(self.linear_size, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, output_dims)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(-1, self.linear_size)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x