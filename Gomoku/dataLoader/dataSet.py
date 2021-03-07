from torchvision import models, transforms
import torchvision
from PIL import Image
import torch
import os
import numpy as np
class GomocupDataset(torch.utils.data.Dataset):
    def __init__(self, path):
        super().__init__()
        data = np.load(path)
        self.states = np.expand_dims(data["states"].reshape(-1, 15, 15), 1).astype(np.float32)
        self.actions = data["actions"].squeeze()
        # print(self.actions.shape)
        # print(self.states.shape)
        
    def __getitem__(self, index):
        return self.states[index], self.actions[index]
    
    def __len__(self):
        return self.actions.shape[0]





def getDataSet(verbose = False):
    phases = ["train", "val", "test"]
    pathes = ["data_set/gomocup/dataset_gomocup15_tfrec_a_one/train_a_one_preprocessed.npz", "data_set/gomocup/dataset_gomocup15_tfrec_a_one/validation_a_one_preprocessed.npz", "data_set/gomocup/dataset_gomocup15_tfrec_a_one/test_a_one_preprocessed.npz"]
    datasets = {}
    for path, phase in zip(pathes, phases):
        datasets[phase] = GomocupDataset(path)
    if verbose:
        for phase in phases:
            print("{} dataset: size {}".format(phase, len(datasets[phase])))
    return datasets, phases

if __name__ == "__main__":
    getDataSet(True)

