import os
import time
import pickle
import json
import tarfile
from glob import glob
from pathlib import Path

from skimage import io
from skimage.color import rgb2gray
from skimage.transform import resize

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision as tv

import numpy as np
import pandas as pd

import matplotlib
import matplotlib.pyplot as plt

from utils.models import DeepTINet
from utils.data_class import TC_images


params = {
    'batch_size': 64, # maskey2020
    'learning_rate': 1e-3,
    'image_channels': 1,
    'num_outputs': 1,
    'train_split': 0.7, # maskey2020
    'num_epochs': 50, # maskey2020
    'print_epoch': 5,
    'save_path': 'deepti/',
    'truncate': 5000, # put in None to run whole dataset
}

dataset = TC_images(
    'data/NASA/',
    'nasa_tropical_storm_competition_train_source',
    'nasa_tropical_storm_competition_train_labels',
    truncate=params["truncate"]
)

print('Dataset info:')
print(f'Number of points: {len(dataset)}')
sample = dataset[0]
print(f'Image shape: {sample["image"].shape}')
print(f'Target shape: {sample["wind_speed"].shape}')
print(f'Data tag: {sample["tag"]} \n')

plt.imshow(sample["image"].squeeze().numpy())
plt.show()

train_set, test_set = torch.utils.data.random_split(dataset, 
    [int(params["train_split"]*len(dataset)), len(dataset) - int(params["train_split"]*len(dataset))])
    
print(f"Training set: {len(train_dataset)}", "\t", f"Number of batches: {len(train_dl)}")
print(f"Testing set: {len(test_dataset)}",   "\t", f"Number of batches: {len(test_dl)}")


# run training loop
train_loss = []
test_loss = []
plot_train = []
plot_valid = []
counter = 0

if not os.path.isdir(params['save_path']):
    os.mkdir(params['save_path'])

for epoch in range(params["num_epochs"]):
    ### Training loop
    model.train()
    running_loss = 0
    plot_train.append(epoch)
    start_time = time.time()
    for iter, batch_sample in enumerate(train_dl):
        x, y = batch_sample["image"].float().to(device), batch_sample["wind_speed"].float().to(device)
        optimizer.zero_grad()
        output = model(x)
        loss = criterion(output, y)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()
        if iter%50 == 0:
            print(f"Epoch {epoch} at batch {iter} loss: {loss.item()}")
    train_loss.append(running_loss/len(train_dl))
    end_time = time.time()
    print(f'Training completed in {round(end_time - start_time,3)} s')
    print()
    start_time = end_time
    
    ### Validation loop
    if epoch%params["print_epoch"] == 0 :
        model.eval()
        running_loss = 0
        plot_valid.append(epoch)
        with torch.no_grad():
            for iter, batch_sample in enumerate(test_dl):
                x, y = batch_sample["image"].float().to(device), batch_sample["wind_speed"].float().to(device)
                output = model(x)
                loss = criterion(output, y)
                running_loss += loss.item()
            test_loss.append(running_loss/len(test_dl))
        if running_loss/len(test_dl) <= test_loss[-1]:
            print(f'Checkpoint saved at epoch {epoch}.')
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': test_loss[-1],
            }, f"{params['save_path']}/model_ckpt.pt")
            counter = 0
        else:
            counter += 1
        
        if counter >= params['patience']:
            print('Early stop. Exiting training.')
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': test_loss[-1],
            }, f"{params['save_path']}/model_ckpt.pt")
            break

        end_time = time.time()
        print()
        print(f'--- Validation at Epoch {epoch} --- ')
        print(f'Training loss: {train_loss[-1]}')
        print(f'Testing loss: {test_loss[-1]}')
        print(f'Validation completed in {round(end_time - start_time,2)} s')
        print('--- End of validation ---')
        print()
        start_time = end_time

print('Training complete!')


# save model and related outputs
PATH = 'deepti.pt'
torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': test_loss[-1],
            }, f"{params['save_path']}/{PATH}")

# save parameters
pickle.dump(
    params, open(f'{params["save_path"]}/parameters_deepti.pkl','wb')
)

# save results
pickle.dump(
    {
    'train_loss':train_loss, 'test_loss': test_loss,
    'plot_train': plot_train, 'plot_test': plot_valid,
    }, open(f'{params["save_path"]}/results_deepti.pkl','wb')
)