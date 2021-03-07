import copy
import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import sklearn.model_selection
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision
from PIL import Image
from sklearn.model_selection import KFold
from torch.autograd import Variable
from torch.optim import lr_scheduler
from torchvision import datasets, models, transforms

from dataLoader import dataSet
from dataLoader import baseManager

datasets, phase = dataSet.getDataSet()


class dataLoaderManager(baseManager.baseManager):
    def __init__(self, batch_size, shuffle, num_workers, drop_last, random_state, pin_memory):
        super().__init__(batch_size, shuffle, num_workers, drop_last, random_state, pin_memory)
        self.dataloaders = {x: torch.utils.data.DataLoader(datasets[x], batch_size=self.batch_size,
                                                            shuffle=self.shuffle, num_workers=self.num_workers, pin_memory=True, 
                                                            drop_last=self.drop_last)
                            for x in phase}
    


if __name__ == "__main__":
    test = dataLoaderManager(batch_size = 16, shuffle = True, 
                                num_workers = 0, drop_last = True, 
                                random_state = 0, pin_memory = True)
    print(test.getDataLoaders())
