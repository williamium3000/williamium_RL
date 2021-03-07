import sys
sys.path.append(".")
from Gomoku import PolicyValueNet
import torch
import torch.nn.functional as F
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
import numpy as np
import torchvision
from torchvision import datasets, models, transforms
import copy
import os
import time
import json
from Gomoku.dataLoader import dataLoaderManager
import logging
def check_accuracy(device, loader, model, phase):
    print('Checking accuracy on %s set: ' % phase)
    logging.warning('Checking accuracy on %s set: ' % phase)
    num_correct = 0
    num_samples = 0
    model.eval()  # set model to evaluation mode
    with torch.no_grad():
        for x, y in loader:
            x = x.to(device=device)  # move to device, e.g. GPU
            y = y.to(device=device)
            actions, value = model(x)
            _, preds = actions.max(1)
            num_correct += (preds == y).sum()
            num_samples += preds.size(0)
        acc = float(num_correct) / num_samples
        print('Got %d / %d correct (%.2f)' % (num_correct, num_samples, 100 * acc))
        logging.warning('Got %d / %d correct (%.2f)' % (num_correct, num_samples, 100 * acc))
        return acc



def train(model, optimizer, is_inception, dataloaders, device, epochs):
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
            if is_inception:
                outputs, aux_outputs = model(x)
                loss1 = F.cross_entropy(outputs, y)
                loss2 = F.cross_entropy(aux_outputs, y)
                loss = loss1 + 0.4 * loss2
            else:
                actions, value = model(x)
                loss = F.cross_entropy(actions, y)
            # Zero out all of the gradients for the variables which the optimizer will update.
            optimizer.zero_grad()

            # This is the backwards pass: compute the gradient of the loss with respect to each parameter of the model.
            loss.backward()

            # Actually update the parameters of the model using the gradients computed by the backwards pass.
            optimizer.step()
        save_model(save_dir = "Gomoku/model_checkpoint", whole_model = False, file_name = "check_point", model = model)

        print('epoche %d, loss = %f' % (e, loss.item()))
        logging.warning('epoche %d, loss = %f' % (e, loss.item()))
        train_acc = check_accuracy(device, dataloaders["train"], model, "train")
        test_acc = check_accuracy(device, dataloaders["val"], model, "validate")
        if test_acc > best_acc:
            best_acc = test_acc
            best_model_wts = copy.deepcopy(model.state_dict())
        
        rec.append((loss.item(), train_acc, test_acc))
        
     
    
    print('Best val Acc: {:4f}'.format(best_acc))
    logging.warning('Best val Acc: {:4f}'.format(best_acc))
    # load best model weights
    model.load_state_dict(best_model_wts)
    save_model(save_dir = "Gomoku/model_checkpoint", whole_model = False, file_name = task_name, model = model)
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
            torch.save(model, save_path + ".pkl")
        else:
            torch.save(model.state_dict(), save_path + ".pkl")
    else:
        print("check point not saved, best_model is None")
        logging.warning("check point not saved, best_model is None")


# configuration
task_name = "SL_alphaGo"
model_name = "alphaGo"
optimizer_name = "Adam"
lr = 0.0001
batch_size = 512
device = torch.device("cuda:3" if torch.cuda.is_available() else "cpu")
num_classes = 255
param_to_update_name_prefix = []
epochs = 200
logging.basicConfig(filename="Gomoku/{}.log".format(task_name))
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
logging.warning("""{}:
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
        epochs))

if __name__ == "__main__":
    

     # load model
    model = PolicyValueNet.PolicyValueNet(15, 15, 1)
    # unfix param
    # for name, param in model.named_parameters():
    #         for i in param_to_update_name_prefix:
    #             if i in name:
    #                 param.requires_grad = True

    # get the param to update
    params_to_update = []
    for name, param in model.named_parameters():
                if param.requires_grad == True:
                    params_to_update.append(param)
    


    # optimizer
    optimizer = getattr(optim, optimizer_name)(params_to_update, lr=lr)
    
    # dataLoaders
    dataLoaders = dataLoaderManager.dataLoaderManager(batch_size = batch_size, 
                                        shuffle = True, 
                                        num_workers = 4, 
                                        drop_last = True, 
                                        random_state = 0, 
                                        pin_memory = True
                                    ).getDataLoaders()
    
    # train and test
    is_inception = False
    if model_name == "inception" or model_name == "googlenet":
        is_inception = True
    best_model, best_acc, rec = train(model = model, 
                                    optimizer = optimizer, 
                                    is_inception = is_inception, 
                                    dataloaders = dataLoaders, 
                                    device = device, 
                                    epochs = epochs
                                    )
    
    
    json.dump(rec, open("rec.json", "w"))
