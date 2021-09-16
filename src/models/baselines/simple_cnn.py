import torch
from torch import nn
import torch.nn.functional as F


class SimpleCNN(nn.Module):
    def __init__(self, num_classes: int = 10, init_weights: bool = True):
        super(SimpleCNN, self).__init__()

        self.conv1 = nn.Conv2d(3, 32, kernel_size=7, padding='same')
        self.conv2 = nn.Conv2d(32, num_classes, kernel_size=1)

        if init_weights:
            self._initialize_weights()

    def _initialize_weights(self) -> None:
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)


    def forward(self, x):

        x = F.relu(self.conv1(x))

        # Make the feature maps width and height = 1, mimics efficient net v2
        x = F.max_pool2d(x, x.shape[-2:])

        out = F.softmax(torch.flatten(self.conv2(x), 1), -1)

        return out