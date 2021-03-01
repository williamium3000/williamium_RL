import torch
from torch import nn
class Q_network(nn.Module):
    def __init__(self, num_obs, num_act):
        super(Q_network, self).__init__()
        self.fc1 = nn.Linear(num_obs, 128, True)
        self.relu1 = nn.ReLU()
        self.fc2 = nn.Linear(128, 128, True)
        self.relu2 = nn.ReLU()
        self.fc3 = nn.Linear(128, num_act, True)
    def forward(self, x):
        x = self.fc1(x)
        x = self.relu1(x)
        x = self.fc2(x)
        x = self.relu2(x)
        x = self.fc3(x)
        return x

if __name__ == "__main__":
    test = Q_network(4, 10)
    t = torch.rand((10, 4))
    print(test(t).size())
