import torch
import torchvision.transforms as T
from torch.utils import data
from torchvision import datasets

import os
from tiny_img import download_tinyImg200

__all__ = ["get_dataloder"]


def get_dataloder(name, batch_size, storage_path):
    trans = T.Compose([T.ToTensor()])
    if name == "CIFAR-100":
        train_set = datasets.CIFAR100(root=storage_path, train=True, download=True,
                                      transform=T.ToTensor())
        test_set = datasets.CIFAR100(root=storage_path, train=False, download=True,
                                      transform=T.ToTensor())
    elif name == "STL10":
        train_set = datasets.STL10(root=storage_path, split="train", download=True,
                                   transform=T.ToTensor())
        test_set = datasets.STL10(root=storage_path, split="test", download=True,
                                      transform=T.ToTensor())
    elif name == "TINY":
        # source: https://colab.research.google.com/github/yandexdataschool/mlhep2019/blob/master/notebooks/day-3/seminar_convnets.ipynb#scrollTo=tvz-gycUrYD1
        if not os.path.exists('./tiny-imagenet-200/'):
            download_tinyImg200('.')
        train_set = datasets.ImageFolder('tiny-imagenet-200/train', transform=T.ToTensor())
        test_set = datasets.ImageFolder('tiny-imagenet-200/val', transform=T.ToTensor())

    return data.DataLoader(train_set, batch_size=batch_size, shuffle=True), \
           data.DataLoader(test_set, batch_size=batch_size, shuffle=False)
