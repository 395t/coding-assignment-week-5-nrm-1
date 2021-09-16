import src
import torch
from torch import nn
from functools import partial


# Put any hyper parameter into your normalization module using partial
norm_mod = partial(nn.BatchNorm2d, 128)
net = src.Backbone(100, norm_mod)
images = torch.rand(16, 3, 256, 256)
labels = torch.randint(high=100, size=(16,))
loss, logits, pred = net(images, labels)
print(f"loss is {loss}, logit shape is {logits.shape}\npredictions {pred}")