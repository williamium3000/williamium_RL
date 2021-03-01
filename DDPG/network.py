import torch
from torch import nn
import numpy as np
import random
def setup_seed(seed):
     torch.manual_seed(seed)
     torch.cuda.manual_seed_all(seed)
     np.random.seed(seed)
     random.seed(seed)
     torch.backends.cudnn.deterministic = True
# 设置随机数种子
setup_seed(20)
class DDPGnetwork(nn.Module):
    def __init__(self, obs_dim, act_dim):
        super(DDPGnetwork, self).__init__()
        self.actor = actorNet(obs_dim, act_dim)
        self.critic = criticNet(obs_dim + act_dim, 1)
    def forward(self, obs):
        return self.actor(obs)
    def value(self, obs, action):
        return self.critic(obs, action)

class actorNet(nn.Module):
    def __init__(self, obs_dim, act_dim):
        super(actorNet, self).__init__()
        self.fc1 = nn.Linear(obs_dim, 32, True)
        self.relu1 = nn.ReLU()
        self.fc2 = nn.Linear(32, 32, True)
        self.relu2 = nn.ReLU()
        self.fc3 = nn.Linear(32, act_dim, True)
        self.tanh3 = nn.Tanh()
    def forward(self, x):
        x = self.fc1(x)
        x = self.relu1(x)
        x = self.fc2(x)
        x = self.relu2(x)
        x = self.fc3(x)
        x = self.tanh3(x)
        return x

class criticNet(nn.Module):
    def __init__(self, obs_dim, act_dim):
        super(criticNet, self).__init__()
        self.fc1 = nn.Linear(obs_dim, 32, True)
        self.relu1 = nn.ReLU()
        self.fc2 = nn.Linear(32, 32, True)
        self.relu2 = nn.ReLU()
        self.fc3 = nn.Linear(32, act_dim, True)
    def forward(self, x, a):
        x = torch.cat([x, a], dim = -1)
        x = self.fc1(x)
        x = self.relu1(x)
        x = self.fc2(x)
        x = self.relu2(x)
        x = self.fc3(x)
        return x

if __name__ == "__main__":
    t = DDPGnetwork(10, 1)
    t1 = DDPGnetwork(10, 1).state_dict()
    t2 = DDPGnetwork(10, 1).state_dict()
    for n, p in t1.items():
        t1[n] = t1[n] - t2[n]
    t.load_state_dict(t1)