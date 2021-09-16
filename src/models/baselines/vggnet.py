import torch
from torch import nn
import torch.nn.functional as F


# Model based off of https://github.com/395t/coding-assignment-week-4-opt-1/blob/main/notebooks/MomentumExperiments.ipynb
class VGGStyleNet(nn.Module):
    def __init__(self, num_classes: int = 10, init_weights: bool = True):
        super(VGGStyleNet, self).__init__()
        self.maxpool = nn.MaxPool2d(1)
        self.maxpool2 = nn.MaxPool2d(2)
        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.conv3_1 = nn.Conv2d(128, 256, kernel_size=3, padding=1)
        self.conv3_2 = nn.Conv2d(256, 256, kernel_size=3, padding=1)
        self.conv4_1 = nn.Conv2d(256, 512, kernel_size=3, padding=1)
        self.conv4_2 = nn.Conv2d(512, 512, kernel_size=3, padding=1)
        self.conv5_1 = nn.Conv2d(512, 512, kernel_size=3, padding=1)
        self.conv5_2 = nn.Conv2d(512, 512, kernel_size=3, padding=1)
        self.classifier = nn.Sequential(
            nn.Linear(512 * 8 * 8, 4096),
            nn.ReLU(True),
            nn.Dropout(),
            nn.Linear(4096, 4096),
            nn.ReLU(True),
            nn.Dropout(),
            nn.Linear(4096, num_classes),
        )
        if init_weights:
            self._initialize_weights()

    def _initialize_weights(self) -> None:
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        out = self.maxpool2(F.relu(self.conv1(x)))
        out = self.maxpool2(F.relu(self.conv2(out)))
        out = F.relu(self.conv3_1(out))
        out = self.maxpool(F.relu(self.conv3_2(out)))
        out = F.relu(self.conv4_1(out))
        out = self.maxpool(F.relu(self.conv4_2(out)))
        out = F.relu(self.conv5_1(out))
        out = self.maxpool(F.relu(self.conv5_2(out)))
        out = torch.flatten(out, 1)
        out = self.classifier(out)
        return out