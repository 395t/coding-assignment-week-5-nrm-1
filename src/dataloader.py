import torch
from torch.utils import data
from torchvision import datasets


__all__ = ["get_dataloder"]


def get_dataloder(name, batch_size):
    if name == "CIFAR-100":
        train_set = datasets.CIFAR100(root="./data", train=True, download=True)
        test_set = datasets.CIFAR100(root="./data", train=False, download=True)
    elif name == "STL10":
        train_set = datasets.STL10(root="./data", split="train", download=True)
        test_set = datasets.STL10(root="./data", split="test", download=True)

    return data.DataLoader(train_set, batch_size=batch_size, shuffle=True), \
           data.DataLoader(test_set, batch_size=batch_size, shuffle=False)