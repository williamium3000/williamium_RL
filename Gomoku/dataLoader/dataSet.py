from torchvision import models, transforms
import torchvision
import yaml
import sklearn.model_selection
from sklearn.model_selection import KFold
import pandas as pd
from PIL import Image
import torch
import os
import numpy as np
class GomocupDataset(torch.utils.data.Dataset):
    def __init__(self, ):
        super().__init__()
        data = np.load("data_set/gomocup/dataset_gomocup15_tfrec_a_one/train_a_one.npz")
        print(data["states"].shape)
        print(data["actions"].shape)
        
    def __getitem__(self, index):
        img_path, label = self.image_pathes[index], self.image_labels[index]
        img = Image.open(img_path)
        if self.transformers:
            img = self.transformers(img)
        return img, label
    
    def __len__(self):
        return len(self.image_pathes)





def getDataSet(verbose = False):
    with open(file = "code/dataLoader/dataSetConfig.yml", mode = "r", encoding = "utf-8") as f:
        random_state = 0

        datasets = None
        configs = yaml.load_all(f.read(), Loader=yaml.FullLoader)
        dataSetConfig = list(configs)[0]
        data_transforms = get_transforms()
        phase = dataSetConfig["phase"]
        if dataSetConfig["dataSetType"] == "SpecificationMappingDataset":
            specificationFilePath = dataSetConfig["specificationFilePath"]
            header = dataSetConfig.get("header", None) # a string list of headers
            test_size_ratio = dataSetConfig["test_size_ratio"]
            if header:
                names = header
                header = 0
            else:
                header = None
                names = None
            specification = pd.read_csv(specificationFilePath, encoding = "utf-8", header = header, names = names) if specificationFilePath.endswith("csv") else pd.read_excel(specificationFilePath, encoding = "utf-8", header = header, names = names)
            image_pathes = list(specification[names[0] if names else 0])
            image_labels = list(specification[names[1] if names else 1])
            assert len(image_pathes) == len(image_labels)
            image_pathes_train, image_pathes_test, image_labels_train, image_labels_test = sklearn.model_selection.train_test_split(image_pathes, image_labels, test_size = test_size_ratio, random_state = random_state, stratify = image_labels)
            pathes_labels = {phase[0]: (image_pathes_train, image_labels_train), phase[1]: (image_pathes_test, image_labels_test)}
            datasets = {x: SpecificationMappingDataset(pathes_labels[x][0], 
                                                        pathes_labels[x][1],
                                                        data_transforms[x])
                            for x in phase}
            
        elif dataSetConfig["dataSetType"] == "ImageFolderDataSet":
            directory = dataSetConfig["directory"]
            datasets = {x: torchvision.datasets.ImageFolder(os.path.join(directory, x),
                                                data_transforms[x])
                        for x in phase}

        if verbose:
            for i in phase:
                print("{} set:".format(i))
                print("number of classes: {}".format(len(datasets[i].classes)))
                print("size: {}".format(len(datasets[i])))
                print("-------------------------------------")


        return datasets, phase

if __name__ == "__main__":
    # getDataSet(True)
    data = np.load("data_set/gomocup/dataset_gomocup15_tfrec_a_one/train_a_one.npz")
    print(data["states"].shape)
    print(data["actions"].shape)
