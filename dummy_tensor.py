from model import DQN
import torch

net = DQN(in_channels=5, num_actions=5)
x = torch.zeros(2, 5, 140, 255)        # batch of 2
y = net(x)
print(y.shape)  # should be torch.Size([2,5])
