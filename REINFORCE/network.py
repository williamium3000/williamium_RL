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
        self.fc = nn.Sequential(
            nn.Linear(num_obs, num_obs, True),
            nn.ReLU(),
            nn.Linear(num_obs, num_act, True),
            nn.Softmax(1)
        )
  

    def forward(self, x):
        return self.fc(x)

if __name__ == "__main__":
    test = Q_network(4, 10)
    t = torch.rand((10, 4))
    print(test(t).size())
