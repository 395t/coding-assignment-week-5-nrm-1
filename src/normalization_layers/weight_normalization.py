import src
import torch
from torch import nn
import torch.nn.functional as F

from functools import partial

from src.paths import CHECKPOINTS_DIR, DATA_DIR
from src.lifecycles import train, test, save_model, load_modal


# An example of a custom normalization layer
class WeightNorm(nn.Module):
    def __init__(self, layer: nn.Module):
        super(WeightNorm, self).__init__()
        self.layer = layer

    def forward(self, x):
        return self.layer(x)


class ConvPoolCNNC(nn.Module):
    def __init__(self, num_classes: int = 100, init_weights: bool = True, normalizer = WeightNorm):
        super(ConvPoolCNNC, self).__init__()

        self.conv1_a = normalizer(nn.Conv2d(3, 96, kernel_size=3, padding="same"))
        self.conv1_b = normalizer(nn.Conv2d(96, 96, kernel_size=3, padding="same"))
        self.conv1_c = normalizer(nn.Conv2d(96, 96, kernel_size=3, padding="same"))
        self.maxpool1 = nn.MaxPool2d(2, stride=2)

        self.conv2_a = normalizer(nn.Conv2d(96, 192, kernel_size=3, padding="same"))
        self.conv2_b = normalizer(nn.Conv2d(192, 192, kernel_size=3, padding="same"))
        self.conv2_c = normalizer(nn.Conv2d(192, 192, kernel_size=3, padding="same"))
        self.maxpool2 = nn.MaxPool2d(2, stride=2)

        self.conv3_a = normalizer(nn.Conv2d(192, 192, kernel_size=3, padding="same"))
        self.conv3_b = normalizer(nn.Conv2d(192, 192, kernel_size=1, padding="same"))
        self.conv3_c = normalizer(nn.Conv2d(192, 192, kernel_size=1, padding="same"))

        self.avgpool1 = nn.AdaptiveAvgPool2d(1)

        self.dense = normalizer(nn.Linear(192, num_classes))

        self.loss = nn.CrossEntropyLoss()

        if init_weights:
            self._initialize_weights()

    def _initialize_weights(self) -> None:
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def forward(self, x, labels_idx):

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

        logits = self.dense(torch.flatten(self.avgpool1(x), 1))

        loss = self.loss(logits, labels_idx)
        pred = torch.argmax(logits, dim=1)
        return loss, logits, pred




if __name__ == "__main__":
    # The goal of the project is to make your own normalization layer according to your paper.
    # A super simple example that does nothing is MyNormLayer

    # This formats the layer such that you can call norm_mod()
    # the partial makes sure to pass 128 into the MyNormLayer(128) later on
    # You can add as many parameters as you need to your layer this way.
    norm_mod = partial(WeightNorm, 128)

    # Create the backbone network with 100 classes and the new MyNormLayer normilization layer
    #net = src.Backbone(100, norm_mod)
    net = ConvPoolCNNC(normalizer=nn.utils.weight_norm)

    # Grab the CIFAR-100 dataset, with a batch size of 10, and store it in the Data Directory (src/data)
    train_dataloader, test_dataloader = src.get_dataloder('CIFAR-100', 10, DATA_DIR)

    # Set up a learning rate and optimizer
    LR = 0.001
    optimizer = torch.optim.Adam(net.parameters(), lr=LR)

    # Train the network on the Adam optimizer, using the training data loader, for 3 epochs
    train(net, optimizer, train_dataloader, epochs=1)

    # Save the model for use later in the checkpoints directory (src/checkpoints) as 'example_model.pt'
    save_model(net, 'weight_norm')

    # load the model from the saved file
    net = load_modal('weight_norm')

    # Test the model on the test dataloader
    test(net, test_dataloader)