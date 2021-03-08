import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np

class res_block(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None, groups=1,
                 base_width=64, dilation=1, norm_layer=None):
        super(res_block, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        if groups != 1 or base_width != 64:
            raise ValueError('BasicBlock only supports groups=1 and base_width=64')
        if dilation > 1:
            raise NotImplementedError("Dilation > 1 not supported in BasicBlock")
        # Both self.conv1 and self.downsample layers downsample the input when stride != 1
        self.conv1 = nn.Conv2d(inplanes, planes, 3, stride, padding=1)
        self.bn1 = norm_layer(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(planes, planes,3, stride, padding=1)
        self.bn2 = norm_layer(planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out



class PolicyValueNet(nn.Module):
    """policy-value network module"""
    def __init__(self, h, w, c):
        super(PolicyValueNet, self).__init__()

        self.h = h
        self.w = w
        # common layers
        self.backbone = nn.Sequential(
            nn.Conv2d(c, 46, 5, 1, 2),
            nn.ReLU(),
            nn.BatchNorm2d(46),
            nn.Conv2d(46, 192, 3, 1, 1),
            nn.ReLU(),
            nn.BatchNorm2d(192),
            self.__build_res_net(4, 192, 192),
            nn.Conv2d(192, 1, 3, 1, 1),
            nn.ReLU(),
        )
        

        # action policy layers
        self.policy_head = nn.Sequential(
            nn.Linear(h * w, h * w),
            nn.Softmax(dim = 1)
        )


        # state value layers
        self.value_head = nn.Sequential(
            nn.Linear(h * w, 128),
            nn.ReLU(),
            nn.Linear(128, 1),
            nn.Tanh()
        )


    def forward(self, states):
        # common layers
        feature = self.backbone(states)
        # action policy layers
        x_act = feature.view(-1, self.h * self.w)
        x_act = self.policy_head(x_act)
        # state value layers
        x_val = feature.view(-1, self.h * self.w)
        x_val = self.value_head(x_val)
        return x_act, x_val
    def __build_res_net(self, num_layers, inplanes, planes):
        layers = []
        for i in range(num_layers):
            layers.append(res_block(inplanes, planes))
        return nn.Sequential(*layers)


if __name__ == "__main__":
    test = PolicyValueNet(15, 15, 1)
    test_tensor = torch.rand((10, 1, 15, 15))
    x_act, x_val = test(test_tensor)
    print(x_act.shape)
    print(x_val.shape)
    loss = torch.mean(x_val - 10)
    loss.backward()
    for p in test.policy_head.parameters():
        print(p.grad)