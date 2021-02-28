import torch
from torch import nn
class PGnetwork(nn.Module):
    def __init__(self, num_obs, num_act):
        super(PGnetwork, self).__init__()
        self.fc1 = nn.Linear(num_obs, 32, True)
        self.bn1 = nn.BatchNorm1d(32)
        self.relu1 = nn.ReLU()
        self.fc2 = nn.Linear(32, 32, True)
        self.bn2 = nn.BatchNorm1d(32)
        self.relu2 = nn.ReLU()
        self.fc3 = nn.Linear(32, num_act, True)
        self.sf = nn.Softmax(1)
    def forward(self, x):
        x = self.fc1(x)
        x = self.bn1(x)
        x = self.relu1(x)
        x = self.fc2(x)
        x = self.bn2(x)
        x = self.relu2(x)
        x = self.fc3(x)
        x = self.sf(x)
        return x

if __name__ == "__main__":
    test = Q_network(4, 10)
    t = torch.rand((10, 4))
    print(test(t).size())
