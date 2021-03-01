import torch
from torch import nn
import random
import numpy as np
def setup_seed(seed):
     torch.manual_seed(seed)
     torch.cuda.manual_seed_all(seed)
     np.random.seed(seed)
     random.seed(seed)
     torch.backends.cudnn.deterministic = True
# 设置随机数种子
setup_seed(20)
class PGnetwork(nn.Module):
    def __init__(self, num_obs, num_act):
        super(PGnetwork, self).__init__()
        self.fc1 = nn.Linear(num_obs, 32, True)
        self.relu1 = nn.ReLU()
        self.fc2 = nn.Linear(32, 32, True)
        self.relu2 = nn.ReLU()
        self.fc3 = nn.Linear(32, num_act, True)
        self.sf = nn.Softmax(1)
    def forward(self, x):
        x = self.fc1(x)
        x = self.relu1(x)
        x = self.fc2(x)
        x = self.relu2(x)
        x = self.fc3(x)
        x = self.sf(x)
        return x

if __name__ == "__main__":
    test = Q_network(4, 10)
    t = torch.rand((10, 4))
    print(test(t).size())
