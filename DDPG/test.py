import torch

obs = torch.tensor([[1, 2, 3, 4]])
a = torch.tensor([[1]])
print(torch.cat([obs, a], dim = -1))