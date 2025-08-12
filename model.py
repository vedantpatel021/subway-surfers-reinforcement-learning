
import torch
import torch.nn as nn
import torch.nn.functional as F
from config import TARGET_H, TARGET_W

class DQN(nn.Module):

    def __init__(self, in_channels: int, num_actions: int):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, 32, kernel_size=8, stride=4)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=4, stride=2)
        self.conv3 = nn.Conv2d(64, 64, kernel_size=3, stride=1)
        
        self.fc1 = nn.Linear(self._feature_size(), 512)
        self.fc2 = nn.Linear(512, num_actions)

    def _feature_size(self):
        dummy = torch.zeros(1, 5, TARGET_H, TARGET_W)
        x = F.relu(self.conv1(dummy))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        return x.view(1, -1).size(1)

    def forward(self, x):
        x = F.relu(self.conv1(x))    # shape: [B,32,?×?]
        x = F.relu(self.conv2(x))    # shape: [B,64,?×?]
        x = F.relu(self.conv3(x))    # shape: [B,64,?×?]
        x = x.view(x.size(0), -1)    # flatten
        x = F.relu(self.fc1(x))      # shape: [B,512]
        return self.fc2(x)           # shape: [B,num_actions]