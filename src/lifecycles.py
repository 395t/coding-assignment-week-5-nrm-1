import torch
from torch.cuda import amp
from src.paths import CHECKPOINTS_DIR, DATA_DIR, STATS_DIR

from tqdm import tqdm
import json

"""
Helper functions for things like training, testing, validating, saving models, loading models, etc. (things you
would do normally in the model testing phase)

Some functions are repurposed from https://github.com/395t/coding-assignment-week-4-opt-1/blob/main/notebooks/MomentumExperiments.ipynb
"""

def save_stats(stats: dict, filename: str):
    with open(str(STATS_DIR / f'{filename}.json'), 'w') as f:
        json.dump(stats, f)

def load_stats(filename: str) -> dict:
    with open(str(STATS_DIR / f'{filename}.json'), 'r') as f:
        return json.load(f)

def save_model(net: torch.nn.Module, name: str):
    torch.save(net, str(CHECKPOINTS_DIR / f'{name}.pt'))


def load_modal(name: str):
    return torch.load(str(CHECKPOINTS_DIR / f'{name}.pt'))


def save_dict(dict, name):
    torch.save(dict, str(CHECKPOINTS_DIR / f'{name}.pt'))


def load_dict(name):
    return torch.load(str(CHECKPOINTS_DIR / f'{name}.pt'))


def get_device():
    return "cuda" if torch.cuda.is_available() else "cpu"


def train(net: torch.nn.Module, optimizer: torch.optim.Optimizer, trainloader, epochs: int = 10,
          starting_epoch: int = 0, loader_description: str = '', gradscaler=None, gpu=None):
    if gpu:
        device = torch.device(gpu)
    else:
        device = get_device()
    net.to(device)
    net.train()

    metrics = {}

    for epoch in range(starting_epoch, epochs):
        correct_images = 0
        total_images = 0
        training_loss = 0

        for batch_index, (images, labels) in enumerate(tqdm(trainloader, desc=loader_description)):
            optimizer.zero_grad()

            images, labels = images.to(device), labels.to(device)

            with amp.autocast(enabled=bool(gradscaler)):
                loss, logits, predicted = net(images, labels)
            if gradscaler:
                gradscaler.scale(loss).backward()
                gradscaler.step(optimizer)
                gradscaler.update()
            else:
                loss.backward()
                optimizer.step()

            training_loss += loss.item()
            total_images += labels.size(0)
            correct_images += predicted.eq(labels).sum().item()

        epoch_metrics = {}
        epoch_metrics['correct_images'] = correct_images
        epoch_metrics['total_images'] = total_images
        epoch_metrics['loss'] = training_loss/(batch_index+1)
        epoch_metrics['accuracy'] = 100.*correct_images/total_images

        metrics[f'epoch_{epoch+1}'] = epoch_metrics

        print('Epoch: %d, Loss: %.3f, '
              'Accuracy: %.3f%% (%d/%d)' % (epoch, training_loss/(batch_index+1),
                                            100.*correct_images/total_images, correct_images, total_images))

    return metrics


def test_validation(net: torch.nn.Module, validloader):
    device = get_device()
    net.to(device)

    val_loss = 0
    total_images = 0
    correct_images = 0

    metrics = {}

    net.eval()
    with torch.no_grad():
        for batch_index, (images, labels) in enumerate(validloader):
            images, labels = images.to(device), labels.to(device)
            loss, logits, predicted = net(images, labels)
            val_loss += loss.item()
            total_images += labels.size(0)
            correct_images += predicted.eq(labels).sum().item()
    val_accuracy = 100.*correct_images/total_images

    metrics['loss'] = val_loss/(batch_index+1)
    metrics['total_images'] = total_images
    metrics['correct_images'] = correct_images
    metrics['accuracy'] = val_accuracy

    #return val_loss/(batch_index+1), val_accuracy
    return metrics



def test(net: torch.nn.Module, testloader, loader_description: str = ''):
    device = get_device()
    net.to(device)

    metrics = {}

    test_loss = 0
    total_images = 0
    correct_images = 0
    net.eval()
    with torch.no_grad():
        for batch_index, (images, labels) in enumerate(tqdm(testloader, desc=loader_description)):
            images, labels = images.to(device), labels.to(device)
            loss, logits, predicted = net(images, labels)
            test_loss += loss.item()
            total_images += labels.size(0)
            correct_images += predicted.eq(labels).sum().item()
    test_accuracy = 100.*correct_images/total_images
    print("Loss on Test Set is", test_loss/(batch_index+1))
    print("Accuracy on Test Set is",test_accuracy)

    metrics['loss'] = test_loss/(batch_index+1)
    metrics['total_images'] = total_images
    metrics['correct_images'] = correct_images
    metrics['accuracy'] = test_accuracy

    return metrics


if __name__ == "__main__":
    import src
    import torch
    from torch import nn
    from functools import partial

    # Put any hyper parameter into your normalization module using partial
    norm_mod = partial(nn.BatchNorm2d, 128)
    net = src.Backbone(100, norm_mod)

    train_dataloader, test_dataloader = src.get_dataloder('CIFAR-100', 10, DATA_DIR)

    LR = 0.001
    optimizer = torch.optim.Adam(net.parameters(), lr=LR)


    train(net, optimizer, train_dataloader, epochs=3)
    # test(net, test_dataloader)
    save_model(net, 'test')