import src
import torch
from torch import nn
from functools import partial

from src.paths import CHECKPOINTS_DIR, DATA_DIR
from src.lifecycles import train, test, save_model, load_modal


# No-op normalization layer
class MyNormLayer(nn.Module):
    def __init__(self, extra_parameter_saved_via_partial):
        super(MyNormLayer, self).__init__()

        # This parameter is passed in via the partial(module, arg1, arg2, arg3, ...) call
        print(extra_parameter_saved_via_partial)

        # more stuff here

    def forward(self, x):
        # more stuff here
        return x


if __name__ == "__main__":
    # Experiments to reproduce from batch norm paper
    experiments = {
        'baseline': {'mod': MyNormLayer, 'lr': 0.001},
        'batch_norm': {'mod': nn.BatchNorm2d, 'lr': 0.001},
        'batch_norm_5x': {'mod': nn.BatchNorm2d, 'lr': 0.005},
        'batch_norm_30x': {'mod': nn.BatchNorm2d, 'lr': 0.030},
    }
    # Grab the CIFAR-100 dataset, with a batch size of 10, and store it in the Data Directory (src/data)
    train_dataloader, test_dataloader = src.get_dataloder('CIFAR-100', 10, DATA_DIR)

    for exp_name, config in experiments.items():
        norm_mod = partial(config['mod'], 128)

        # Create the backbone network with 100 classes and the new MyNormLayer normilization layer
        net = src.Backbone(100, norm_mod)

        # Set up a learning rate and optimizer
        LR = config['lr']
        optimizer = torch.optim.Adam(net.parameters(), lr=LR)

        # Train the network on the Adam optimizer, using the training data loader, for 10 epochs
        train(net, optimizer, train_dataloader, epochs=10)

        # Save the model for use later in the checkpoints directory (src/checkpoints) as 'example_model.pt'
        save_model(net, exp_name)

        # load the model from the saved file
        net = load_modal(exp_name)

        # Test the model on the test dataloader
        test(net, test_dataloader)
