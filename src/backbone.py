import torch
from torch import nn


__all__ = ["Backbone"]


class ResMod(nn.Module):
    def __init__(self, channels, norm_mod):
        super(ResMod, self).__init__()
        self.stem = nn.Sequential(
            nn.Conv2d(channels, channels//2, 3, padding="same"),
            nn.Conv2d(channels//2, channels, 3, padding="same"), norm_mod(), nn.ReLU())

    def forward(self, x):
        return self.stem(x) + x


class Backbone(nn.Module):
    def __init__(self, num_classes, norm_mod):
        super(Backbone, self).__init__()
        # num_channels should be constant
        num_channels = 128
        self.init_conv = nn.Sequential(nn.Conv2d(3, 128, 3, padding="same"), norm_mod(), nn.ReLU())
        self.stage1 = nn.Sequential(ResMod(num_channels, norm_mod), ResMod(num_channels, norm_mod),
                                    nn.MaxPool2d(2))
        self.stage2 = nn.Sequential(ResMod(num_channels, norm_mod), ResMod(num_channels, norm_mod),
                                    nn.MaxPool2d(2))
        self.stage3 = ResMod(num_channels, norm_mod)
        self.pool = nn.AdaptiveMaxPool2d(1)
        self.logits = nn.Linear(num_channels, num_classes)
        self.loss = nn.CrossEntropyLoss()

    def forward(self, img_batch, labels_idx):
        # Batch size must NOT be 1
        feature_map = self.stage3(self.stage2(self.stage1(self.init_conv(img_batch))))
        logits = self.logits(torch.squeeze(self.pool(feature_map)))
        loss = self.loss(logits, labels_idx)
        pred = torch.argmax(logits, dim=1)
        return loss, logits, pred
