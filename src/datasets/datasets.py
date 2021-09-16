import torch
import torchvision
import torchvision.transforms as transforms

"""
This file holds functions for fetching train, test, validation data loaders for different datasets.

Some functions are reused from https://github.com/395t/coding-assignment-week-4-opt-1/blob/main/notebooks/MomentumExperiments.ipynb
"""

_transform = transforms.Compose(
    [transforms.ToTensor(),
     transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
transform_train = transforms.Compose([
    transforms.RandomCrop(32, padding=4),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
])

transform_test = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
])

generator=torch.Generator().manual_seed(42)


def get_cifar_10(batch_size: int = 64):
    trainset = torchvision.datasets.CIFAR10(root='./data', train=True,
                                          download=True, transform=transform_train)
    testset = torchvision.datasets.CIFAR10(root='./data', train=False,
                                          download=True, transform=transform_test)
    trainset, validset = torch.utils.data.random_split(trainset,
                                                      [int(len(trainset)*0.8),len(trainset)-
                                                      int(len(trainset)*0.8)], generator=generator)

    trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size,
                                              shuffle=True, num_workers=2)
    validloader = torch.utils.data.DataLoader(validset, batch_size=batch_size,
                                              shuffle=False, num_workers=2)
    testloader = torch.utils.data.DataLoader(testset, batch_size=batch_size,
                                             shuffle=False, num_workers=2)

    return trainloader, testloader, validloader