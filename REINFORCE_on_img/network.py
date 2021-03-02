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

        self.features = (w // 2) * (h // 2) * 32

        self.block1 = nn.Sequential(
            nn.Conv2d(c, 32, 3, 1, 1),
            nn.ReLU(),
            nn.Conv2d(32, 32, 3, 1, 1),
            nn.ReLU(),
            nn.Conv2d(32, 32, 3, 1, 1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2)
        )

        self.fc = nn.Sequential(
            nn.Linear(self.features, 32),
            nn.ReLU(),
            nn.Linear(32, num_act),
            nn.Softmax(1)
            )


        
    def forward(self, x):
        x = self.block1(x)
        x = x.reshape(x.shape[0], -1)
        x = self.fc(x)
        return x

if __name__ == "__main__":
    test = PGnetwork(32, 32, 3, 10)
    t = torch.rand((10, 3, 32, 32))
    print(test(t).size())
