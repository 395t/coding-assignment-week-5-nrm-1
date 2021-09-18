import src
import torch
from torch import nn
import torch.nn.functional as F

from functools import partial

from src.paths import CHECKPOINTS_DIR, DATA_DIR
from src.lifecycles import train, test, save_model, load_modal, save_stats, load_stats
from src.viz_helper import compare_training_stats, save_plt



# An example of a custom normalization layer
class WeightNorm(nn.Module):
    def __init__(self, layer: nn.Module):
        super(WeightNorm, self).__init__()
        self.layer = layer

    def forward(self, x):
        #TODO implement my own weight norm
        return self.layer(x)


class WNResMod(nn.Module):
    def __init__(self, channels, norm_mod):
        super(WNResMod, self).__init__()
        self.stem = nn.Sequential(
            norm_mod(nn.Conv2d(channels, channels//2, 3, padding="same")),
            norm_mod(nn.Conv2d(channels//2, channels, 3, padding="same")), nn.ReLU())

    def forward(self, x):
        return self.stem(x) + x


class WNBackbone(nn.Module):
    def __init__(self, num_classes, norm_mod):
        super(WNBackbone, self).__init__()
        # num_channels should be constant
        num_channels = 128
        self.init_conv = nn.Sequential(norm_mod(nn.Conv2d(3, 128, 3, padding="same")), nn.ReLU())
        self.stage1 = nn.Sequential(WNResMod(num_channels, norm_mod), WNResMod(num_channels, norm_mod),
                                    nn.MaxPool2d(2))
        self.stage2 = nn.Sequential(WNResMod(num_channels, norm_mod), WNResMod(num_channels, norm_mod),
                                    nn.MaxPool2d(2))
        self.stage3 = WNResMod(num_channels, norm_mod)
        self.pool = nn.AdaptiveMaxPool2d(1)
        self.logits = norm_mod(nn.Linear(num_channels, num_classes))
        self.loss = nn.CrossEntropyLoss()

    def forward(self, img_batch, labels_idx):
        # Batch size must NOT be 1
        feature_map = self.stage3(self.stage2(self.stage1(self.init_conv(img_batch))))
        logits = self.logits(torch.squeeze(self.pool(feature_map)))
        loss = self.loss(logits, labels_idx)
        pred = torch.argmax(logits, dim=1)
        return loss, logits, pred


class ConvPoolCNNC(nn.Module):
    def __init__(self, normalizer, num_classes: int = 100, init_weights: bool = True):
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
    EPOCHS = 1
    BATCH_SIZE = 64

    net_cpc_w_norm = ConvPoolCNNC(normalizer=nn.utils.weight_norm)
    net_cpc_wo_norm = ConvPoolCNNC(normalizer=nn.Sequential)

    # I copied the backbone and modified it slightly to work for my paper (I have to have access to the layer,
    # others may not need this though).
    net_bb_w_norm = WNBackbone(100, norm_mod=nn.utils.weight_norm)
    net_bb_wo_norm = WNBackbone(100, norm_mod=nn.Sequential)

    # Different models with some parameters I want to compare against
    configs = [
        {'name': 'Conv Pool C with Weight Norm', 'model': net_cpc_w_norm, 'save_model': 'net_cpc_w_norm', 'save_stats': 'net_cpc_w_norm_training'},
        {'name': 'Conv Pool C with out Weight Norm', 'model': net_cpc_wo_norm, 'save_model': 'net_cpc_wo_norm', 'save_stats': 'net_cpc_wo_norm_training'},
        {'name': 'Backbone with Weight Norm', 'model': net_bb_w_norm, 'save_model': 'net_bb_w_norm', 'save_stats': 'net_bb_w_norm_training'},
        {'name': 'Backbone with out Weight Norm', 'model': net_bb_wo_norm, 'save_model': 'net_bb_wo_norm', 'save_stats': 'net_bb_wo_norm_training'}
    ]

    # Train each model
    for config in configs:
        net = config['model']

        # Grab the CIFAR-100 dataset, with a batch size of 10, and store it in the Data Directory (src/data)
        train_dataloader, test_dataloader = src.get_dataloder('CIFAR-100', BATCH_SIZE, DATA_DIR)

        # Set up a learning rate and optimizer
        LR = 0.01
        optimizer = torch.optim.Adam(net.parameters(), lr=LR)

        # Train the network on the Adam optimizer, using the training data loader, for 3 epochs
        stats = train(net, optimizer, train_dataloader, epochs=EPOCHS, loader_description=config['name'])

        # Save the model for testing later
        save_model(net, config['save_model'])
        # Save the stats from the training loop for later
        save_stats(stats, config['save_stats'])

    # Models have run, lets plot the stats
    all_stats = []
    labels = []
    for config in configs:
        all_stats.append(load_stats(config['save_stats']))
        labels.append(config['name'])

    # For every config, plot the loss across number of epochs
    plt = compare_training_stats(all_stats, labels)
    save_plt(plt, 'loss_test')
    # plt.show WILL WIPE THE PLT, so make sure you save the plot before you show it
    plt.show()

    # For every config, plot the accuracy across the number of epochs
    plt = compare_training_stats(all_stats, labels, metric_to_compare='accuracy', y_label='accuracy', title='Accuracy vs Epoch')
    save_plt(plt, 'acc_test')
    # plt.show WILL WIPE THE PLT, so make sure you save the plot before you show it
    plt.show()


