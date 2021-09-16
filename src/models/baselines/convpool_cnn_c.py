import torch
from torch import nn
import torch.nn.functional as F


class ConvPoolCNNC(nn.Module):
    def __init__(self, num_classes: int = 10, init_weights: bool = True):
        super(ConvPoolCNNC, self).__init__()

        self.conv1_a = nn.Conv2d(3, 96, kernel_size=3, padding="same")
        self.conv1_b = nn.Conv2d(96, 96, kernel_size=3, padding="same")
        self.conv1_c = nn.Conv2d(96, 96, kernel_size=3, padding="same")
        self.maxpool1 = nn.MaxPool2d(2, stride=2)

        self.conv2_a = nn.Conv2d(96, 192, kernel_size=3, padding="same")
        self.conv2_b = nn.Conv2d(192, 192, kernel_size=3, padding="same")
        self.conv2_c = nn.Conv2d(192, 192, kernel_size=3, padding="same")
        self.maxpool2 = nn.MaxPool2d(2, stride=2)

        self.conv3_a = nn.Conv2d(192, 192, kernel_size=3, padding="same")
        self.conv3_b = nn.Conv2d(192, 192, kernel_size=1, padding="same")
        self.conv3_c = nn.Conv2d(192, num_classes, kernel_size=1, padding="same")


        if init_weights:
            self._initialize_weights()

    def _initialize_weights(self) -> None:
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)


    def forward(self, x):

        x = F.leaky_relu(self.conv1_a(x))
        x = F.leaky_relu(self.conv1_b(x))
        x = F.leaky_relu(self.conv1_c(x))
        x = self.maxpool1(x)

        x = F.dropout(x)

        x = F.leaky_relu(self.conv2_a(x))
        x = F.leaky_relu(self.conv2_b(x))
        x = F.leaky_relu(self.conv2_c(x))
        x = self.maxpool2(x)

        x = F.dropout(x)

        x = F.leaky_relu(self.conv3_a(x))
        x = F.leaky_relu(self.conv3_b(x))
        x = F.leaky_relu(self.conv3_c(x))

        out = F.softmax(torch.flatten(F.avg_pool2d(x, x.shape[-2:]), 1), 1)



        return out