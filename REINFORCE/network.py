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
class ravel(nn.Module):
    def __init__(self):
        super(ravel, self).__init__()
    def forward(self, x):
        return x.view(x.shape[0], -1)
class PGnetwork(nn.Module):
    def __init__(self, num_obs, num_act):
        super(PGnetwork, self).__init__()
        features = (num_obs[0] // 8) * (num_obs[1] // 8) * 64
        # print(features)
        self.layers = nn.Sequential(
            # nn.Linear(num_obs, num_obs, True),
            # nn.ReLU(),
            # nn.Linear(num_obs, 64, True),
            # nn.ReLU(),
            # nn.Linear(64, num_act, True),
            # nn.Softmax(1)

            nn.Conv2d(num_obs[2], 32, 7, stride=1, padding=3),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(32, 64, 5, stride=1, padding=2),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(64, 64, 3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            # nn.Conv2d(64, 64, 3, stride=1, padding=1),
            # nn.ReLU(),
            # nn.MaxPool2d(2, 2),
            ravel(),
            nn.Linear(features, 64, True),
            nn.ReLU(),
            nn.Linear(64, num_act, True),
            nn.Softmax(1)

        )

    def forward(self, x):
        return self.layers(x)

if __name__ == "__main__":
    test = PGnetwork((105, 80, 3), 10)
    t = torch.rand((10, 3, 105, 80))
    print(test(t).size())
