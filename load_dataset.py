import torchvision
import torch
import numpy as np
from torch.utils.data import TensorDataset, DataLoader
from torchvision.transforms import transforms



def preprocess_data(train_data, valid_split, batch):
    x_train_data = []
    y_train_data = []
    for x, y in train_data:
        nc = np.array(x)
        x_train_data.append(nc.tolist())
        y_train_data.append([y])
    temp_dataloader_valid = None
    temp_dataloader_train = None
    if valid_split == False:
        x_data = torch.tensor(x_train_data)
        y_data = torch.tensor(y_train_data)
        temp_dataset = TensorDataset(x_data, y_data)
        temp_dataloader_train = DataLoader(temp_dataset, shuffle=True, batch_size=batch)
    else:
        length = int(len(x_train_data)*0.9)
        x_train, x_valid = x_train_data[:length], x_train_data[length:]
        y_train, y_valid = y_train_data[:length], y_train_data[length:]
        x_data_train, x_data_valid = torch.tensor(x_train), torch.tensor(x_valid)
        y_data_train, y_data_valid = torch.tensor(y_train), torch.tensor(y_valid)
        temp_dataset_train = TensorDataset(x_data_train, y_data_train)
        temp_dataset_valid = TensorDataset(x_data_valid, y_data_valid)
        temp_dataloader_train = DataLoader(temp_dataset_train, shuffle=True, batch_size=batch)
        temp_dataloader_valid = DataLoader(temp_dataset_valid, shuffle=True, batch_size=batch)
    return temp_dataloader_train, temp_dataloader_valid


def get_data_loaders(download = False, split='train', batch=32):
    traget_data = None
    valid_split = False
    train_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])
    if split=='train':
        traget_data = torchvision.datasets.CIFAR10(root='./data_folder/', train=True, download=download, transform=train_transform)
        valid_split = True
    else:
        traget_data = torchvision.datasets.CIFAR10(root='./data_folder/', train=False, download=download, transform=transform)
    traget_dataloader, test_dataloader = preprocess_data(traget_data, valid_split, batch)
    return traget_dataloader, test_dataloader
