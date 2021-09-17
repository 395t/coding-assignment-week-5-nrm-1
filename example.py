import src
import torch
from torch import nn
from functools import partial

from src.paths import CHECKPOINTS_DIR, DATA_DIR
from src.lifecycles import train, test, save_model, load_modal

def basic_backbone_example():
    # Put any hyper parameter into your normalization module using partial
    norm_mod = partial(nn.BatchNorm2d, 128)
    net = src.Backbone(100, norm_mod)
    images = torch.rand(16, 3, 256, 256)
    labels = torch.randint(high=100, size=(16,))
    loss, logits, pred = net(images, labels)
    print(f"loss is {loss}, logit shape is {logits.shape}\npredictions {pred}")


# An example of a custom normalization layer
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
    # The goal of the project is to make your own normalization layer according to your paper.
    # A super simple example that does nothing is MyNormLayer

    # This formats the layer such that you can call norm_mod()
    # the partial makes sure to pass 128 into the MyNormLayer(128) later on
    # You can add as many parameters as you need to your layer this way.
    norm_mod = partial(MyNormLayer, 128)

    # Create the backbone network with 100 classes and the new MyNormLayer normilization layer
    net = src.Backbone(100, norm_mod)

    # Grab the CIFAR-100 dataset, with a batch size of 10, and store it in the Data Directory (src/data)
    train_dataloader, test_dataloader = src.get_dataloder('CIFAR-100', 10, DATA_DIR)

    # Set up a learning rate and optimizer
    LR = 0.001
    optimizer = torch.optim.Adam(net.parameters(), lr=LR)

    # Train the network on the Adam optimizer, using the training data loader, for 3 epochs
    train(net, optimizer, train_dataloader, epochs=3)

    # Save the model for use later in the checkpoints directory (src/checkpoints) as 'example_model.pt'
    save_model(net, 'example_model')

    # load the model from the saved file
    net = load_modal('example_model')

    # Test the model on the test dataloader
    test(net, test_dataloader)