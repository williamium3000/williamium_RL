import sys
sys.path.append("REINFORCE_on_img")
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
import time
import json
import network
def check_accuracy(device, loader, model, phase):
    print('Checking accuracy on %s set: ' % phase)
    num_correct = 0
    num_samples = 0
    model.eval()  # set model to evaluation mode
    with torch.no_grad():
        for x, y in loader:
            x = x.to(device=device)  # move to device, e.g. GPU
            y = y.to(device=device)
            scores = model(x)
            _, preds = scores.max(1)
            num_correct += (preds == y).sum()
            num_samples += preds.size(0)
        acc = float(num_correct) / num_samples
        print('Got %d / %d correct (%.2f)' % (num_correct, num_samples, 100 * acc))
        return acc



def train(task_name, model, optimizer, dataloaders, device, epochs):
    """
    Inputs:
        - model: A PyTorch model to train.
        - optimizer: An Optimizer object we will use to train the model.
        - is_inception: True if the model is inception or googlenet.
        - dataloaders: dataLoaders of the data.
        - device: gpu device(or cpu) on which the training process is on.
        - epochs: the total number of epochs to train.
    Returns: 
        - model: Model with best test acc
        - best_acc: the val accuracy of the best model
        - rec: the information of the training process
            - loss of each epochs
            - train acc of each epochs
            - val acc of each epochs
    """
    rec = []
    model = model.to(device=device)  # move the model parameters to CPU/GPU
    best_model_wts = copy.deepcopy(model.state_dict())
    best_acc = 0.0
    for e in range(epochs):
        for t, (x, y) in enumerate(dataloaders["train"]):
            model.train()  # put model to training mode
            x = x.to(device=device)  # move to device, e.g. GPU
            y = y.to(device=device)

            outputs = model(x)
            loss = F.cross_entropy(outputs, y)
            # print(loss.item())
            # Zero out all of the gradients for the variables which the optimizer will update.
            optimizer.zero_grad()

            # This is the backwards pass: compute the gradient of the loss with respect to each parameter of the model.
            loss.backward()

            # Actually update the parameters of the model using the gradients computed by the backwards pass.
            optimizer.step()

        print('epoche %d, loss = %f' % (e, loss.item()))
        train_acc = check_accuracy(device, dataloaders["train"], model, "train")
        test_acc = check_accuracy(device, dataloaders["val"], model, "validate")
        if test_acc > best_acc:
            best_acc = test_acc
            best_model_wts = copy.deepcopy(model.state_dict())
        
        rec.append((loss.item(), train_acc, test_acc))
        
     
    
    print('Best val Acc: {:4f}'.format(best_acc))
    # load best model weights
    model.load_state_dict(best_model_wts)
    save_model(save_dir = "REINFORCE_on_img", whole_model = False, file_name = task_name, model = model)
    return model, best_acc, rec

def save_model(save_dir, whole_model, file_name = None, model = None):
    if not os.path.exists(save_dir):
        os.mkdir(save_dir)
    if file_name:
        save_path = os.path.join(save_dir, file_name)
    else:
        save_path = os.path.join(save_dir, str(int(time.time())))
    if model:
        if whole_model:
            torch.save(model, save_path + ".pth")
        else:
            torch.save(model.state_dict(), save_path + ".pth")
    else:
        print("check point not saved, best_model is None")


# configuration
task_name = "Pong-v0"
model_name = "customized"
optimizer_name = "Adam"
lr = 0.0001
batch_size = 512
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
num_classes = 100
param_to_update_name_prefix = []
epochs = 500
print(
    """{}:
    - model name: {}
    - optimizer: {}
    - learning rate: {}
    - batch size: {}
    - device : {}
    - num_of_classes: {}
    - param_to_update_name_prefix: {}
    - epochs: {}
 """.format(
        task_name, 
        model_name, 
        optimizer_name, 
        lr, 
        batch_size,
        device, 
        num_classes, 
        param_to_update_name_prefix, 
        epochs)
)

if __name__ == "__main__":
    data_transforms = {
            'train': transforms.Compose([
                transforms.RandomRotation(degrees = 50),
                # transforms.RandomResizedCrop((224, 224)),
                # transforms.Resize((size, size)),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.RandomErasing(p=0.2, scale=(0.02, 0.33), ratio=(0.3, 3.3), value= "random", inplace=False),
                transforms.Normalize(0, 1)
            ]),
            'val': transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize(0, 1)
            ]),
        }
    CIFAR10_train_dataSet = datasets.CIFAR100("data_set/cifar100", train = True, transform = data_transforms["train"], download=True)
    CIFAR10_val_dataSet = datasets.CIFAR100("data_set/cifar100", train = False, transform = data_transforms["val"], download=True)
    CIFAR10_train = torch.utils.data.DataLoader(CIFAR10_train_dataSet,
                                          batch_size=batch_size,
                                          shuffle=True,
                                          num_workers=8,
                                          pin_memory=True)
    CIFAR10_val = torch.utils.data.DataLoader(CIFAR10_val_dataSet,
                                          batch_size=batch_size,
                                          shuffle=True,
                                          num_workers=8,
                                          pin_memory=True)

     # load model
    model = network.PGnetwork(w = 32, h = 32, c = 3, num_act = num_classes)
    model.load_state_dict(torch.load("REINFORCE_on_img/Pong-v0.pth"))
    # unfix param
    for name, param in model.named_parameters():
            for i in param_to_update_name_prefix:
                if i in name:
                    param.requires_grad = True

    # get the param to update
    params_to_update = []
    for name, param in model.named_parameters():
                if param.requires_grad == True:
                    params_to_update.append(param)
    # # params to update
    # print(params_to_update)
    # # display layers
    # print(model.modules())
    # for m in model.modules():
    #     print(m)

    # optimizer
    optimizer = getattr(optim, optimizer_name)(params_to_update, lr=lr)
    
    # dataLoaders
    dataLoaders = {"train":CIFAR10_train, "val":CIFAR10_val}
    
    # train and test
    best_model, best_acc, rec = train(task_name = task_name,
                                    model = model, 
                                    optimizer = optimizer, 
                                    dataloaders = dataLoaders, 
                                    device = device, 
                                    epochs = epochs
                                    )
    
    
    json.dump(rec, open("REINFORCE_on_img/pretrain.json", "w"))
