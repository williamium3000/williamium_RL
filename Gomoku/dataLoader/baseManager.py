import torch
import torch.nn.functional as F
from torch.autograd import Variable
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
import numpy as np
import torchvision
from torchvision import datasets, models, transforms
import matplotlib.pyplot as plt
import copy
import os
import abc
class baseManager(metaclass=abc.ABCMeta):
    def __init__(self, batch_size, shuffle, num_workers, drop_last, random_state, pin_memory):
        super().__init__()
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.num_workers = num_workers
        self.drop_last = drop_last 
        self.random_state = random_state
        self.pin_memory = pin_memory
        self.dataloaders = None
        
    def getDataLoaders(self):
        return self.dataloaders
                
    
if __name__ == "__main__":
    pass