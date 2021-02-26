import torch
import numpy as np
test = torch.rand((32, 2))
idx = torch.ones((32, 1))
print(torch.gather(test, 1, idx))