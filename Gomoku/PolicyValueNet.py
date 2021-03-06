import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np

class PolicyValueNet(nn.Module):
    """policy-value network module"""
    def __init__(self, h, w, c):
        super(Net, self).__init__()

        self.h = h
        self.w = w
        # common layers
        self.backbone = nn.Sequential(
            nn.Conv2d(c, 32, kernel_size=5, padding=2),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(128, 1, kernel_size=1),
            nn.ReLU(),
        )
        


        # action policy layers
        self.policy_head = nn.Sequential(
            nn.Linear(h * w, h * w)
            nn.Softmax(dim = 1)
        )


        # state value layers
        self.value_head = nn.Sequential(
            nn.Linear(h * w, 64),
            nn.ReLU(),
            nn.Linear(64, 1),
            nn.Tanh()
        )


    def forward(self, states):
        # common layers
        feature = self.backbone(states)
        # action policy layers
        x_act = feature.view(-1, h * w)
        x_act = self.policy_head(x_act)
        # state value layers
        x_val = feature.view(-1, h * w)
        x_val = self.value_head(x_val)
        return x_act, x_val
