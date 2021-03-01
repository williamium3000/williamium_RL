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
    def __init__(self, w, h, c, num_act):
        super(PGnetwork, self).__init__()

        self.features = (w // 4) * (h // 4) * 128

        self.block1 = nn.Sequential(
            nn.Conv2d(c, 64, 3, 1, 1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Conv2d(64, 64, 3, 1, 1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Conv2d(64, 64, 3, 1, 1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2, 2)
        )
        self.block2 = nn.Sequential(
            nn.Conv2d(64, 128, 3, 1, 1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.Conv2d(128, 128, 3, 1, 1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.Conv2d(128, 128, 3, 1, 1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(2, 2)
        )
        self.fc = nn.Sequential(
            nn.Linear(self.features, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.Linear(32, num_act),
            # nn.Softmax(1)
            )


        
    def forward(self, x):
        x = self.block1(x)
        x = self.block2(x)
        x = x.reshape(x.shape[0], -1)
        x = self.fc(x)
        return x

if __name__ == "__main__":
    test = PGnetwork(32, 32, 3, 10)
    t = torch.rand((10, 3, 32, 32))
    print(test(t).size())
