import src
import torch
from torch import nn
from functools import partial

from src.paths import CHECKPOINTS_DIR, DATA_DIR
from src.lifecycles import train, test, save_model, load_modal, save_stats, load_stats
from src.viz_helper import compare_training_stats, save_plt

def basic_backbone_example():
    # Put any hyper parameter into your normalization module using partial
    norm_mod = partial(nn.BatchNorm2d, 128)
    net = src.Backbone(100, norm_mod)
    images = torch.rand(16, 3, 256, 256)
    labels = torch.randint(high=100, size=(16,))
    loss, logits, pred = net(images, labels)
    print(f"loss is {loss}, logit shape is {logits.shape}\npredictions {pred}")


# An example of a custom normalization layer - dropout
class Dropout(nn.Module):
    def __init__(self, extra_parameter_saved_via_partial, p=0.0):
        super(Dropout, self).__init__()
        # Using pytorch built-in Dropout module 
        # This nn module applies dropout only during training
        # -> no need to add condition to turn dropout on/off during training/testing
        self.drop_layer = nn.Dropout(p=p)

    def forward(self, x):
        # Simply apply dropout, only during training
        x = self.drop_layer(x)
        # matrix that is the same size as x | d
        # d have zeros and ones, where the 1s are populated p percent of the time
        return x


if __name__ == "__main__":
    # This formats the layer such that you can call norm_mod()
    # the partial makes sure to pass 128 into the MyNormLayer(128) later on
    # You can add as many parameters as you need to your layer this way.

    #Create dropout normalizers with p = 0.0, p = 0.25, p = 0.5, and p = 0.75
    norm_mod_00 = partial(Dropout, 128, 0.0)
    norm_mod_00_a = partial(Dropout, 128, 0.0)

    norm_mod_25 = partial(Dropout, 128, 0.25)
    norm_mod_25_a = partial(Dropout, 128, 0.25)

    norm_mod_50 = partial(Dropout, 128, 0.50)
    norm_mod_50_a = partial(Dropout, 128, 0.50)
    norm_mod_50_b = partial(Dropout, 128, 0.50)
    norm_mod_50_c = partial(Dropout, 128, 0.50)

    norm_mod_75 = partial(Dropout, 128, 0.75)
    norm_mod_75_a = partial(Dropout, 128, 0.75)

    # Create the backbone networks with 100 classes and their respective dropout modules
    # Creating two models to accomodate the two different LRs to be tested per model
    # Additional p = 0.50 model with LR = 0.007 for global comparison
    net_no_dropout_lr1 = src.Backbone(100, norm_mod_00)
    net_no_dropout_lr2 = src.Backbone(100, norm_mod_00_a)

    net_dropout_25_lr1 = src.Backbone(100, norm_mod_25)
    net_dropout_25_lr2 = src.Backbone(100, norm_mod_25_a)

    net_dropout_50_lr1 = src.Backbone(100, norm_mod_50)
    net_dropout_50_lr2 = src.Backbone(100, norm_mod_50_a)
    net_dropout_50_003 = src.Backbone(100, norm_mod_50_b)
    net_dropout_50_007 = src.Backbone(100, norm_mod_50_c)

    net_dropout_75_lr1 = src.Backbone(100, norm_mod_75)
    net_dropout_75_lr2 = src.Backbone(100, norm_mod_75_a)

    # Different models using different dropout probabilities
    configs = [
        # no dropout
        {'name': 'No dropout, LR: 0.001', 'model': net_no_dropout_lr1, 'save_model': 'net_no_dropout_lr1', 'save_stats': 'net_no_dropout_training_lr1', 'LR': 0.001},
        {'name': 'No dropout, LR: 0.01', 'model': net_no_dropout_lr2, 'save_model': 'net_no_dropout_lr2', 'save_stats': 'net_no_dropout_training_lr2', 'LR': 0.01},
        
        # dropout p = 0.25
        {'name': 'Dropoiut of 25%, LR: 0.001', 'model': net_dropout_25_lr1, 'save_model': 'net_dropout_25_lr1', 'save_stats': 'net_dropout_25_training_lr1', 'LR': 0.001},
        {'name': 'Dropoiut of 25%, LR: 0.01', 'model': net_dropout_25_lr2, 'save_model': 'net_dropout_25_lr2', 'save_stats': 'net_dropout_25_training_lr2', 'LR': 0.01},

        # dropout p = 0.50
        {'name': 'Dropout of 50%, LR: 0.001', 'model': net_dropout_50_lr1, 'save_model': 'net_dropout_50_lr1', 'save_stats': 'net_dropout_50_lr1', 'LR': 0.001},
        {'name': 'Dropout of 50%, LR: 0.01', 'model': net_dropout_50_lr2, 'save_model': 'net_dropout_50_lr2', 'save_stats': 'net_dropout_50_lr2', 'LR': 0.01},
        {'name': 'Dropout of 50%, LR: 0.003', 'model': net_dropout_50_003, 'save_model': 'net_dropout_50_003', 'save_stats': 'net_dropout_50_003', 'LR': 0.003},
        {'name': 'Dropout of 50%, LR: 0.007', 'model': net_dropout_50_007, 'save_model': 'net_dropout_50_007', 'save_stats': 'net_dropout_50_007', 'LR': 0.007},


        # dropout p = 0.75
        {'name': 'Dropout of 75%, LR: 0.001', 'model': net_dropout_75_lr1, 'save_model': 'net_dropout_75_lr1', 'save_stats': 'net_dropout_75_lr1', 'LR': 0.001},
        {'name': 'Dropout of 75%, LR: 0.01', 'model': net_dropout_75_lr2, 'save_model': 'net_dropout_75_lr2', 'save_stats': 'net_dropout_75_lr2', 'LR': 0.01}
    ]

    # Train the network on the Adam optimizer, using the training data loader, for 3 epochs
    # This will train the four different dropout configurations, each with LR of 0.001, and 0.01
    test_data = []
    for config in configs:
        net = config['model']

        # Grab the CIFAR-100 dataset, with a batch size of 10, and store it in the Data Directory (src/data)
        train_dataloader, test_dataloader = src.get_dataloder('CIFAR-100', 64, DATA_DIR)

        # Set up a learning rate and optimizer
        optimizer = torch.optim.Adam(net.parameters(), lr=config['LR'])

        # Train network
        stats = train(net, optimizer, train_dataloader, epochs=20, loader_description=config['name'])

        # Save the model for use later in the checkpoints directory (src/checkpoints) as 'example_model.pt'
        save_model(net, config['save_model'])

        # Save the stats from the training loop for later
        save_stats(stats, config['save_stats'])

        test_stats = test(net, test_dataloader)
        test_data.append({'name': config['name'], 'acc': test_stats['accuracy']})

        print(f"TESTING COMPLETE: {config['name']}: {test_stats['accuracy']}")

        save_stats(test_stats, f"dropout_test_{config['save_stats']}")

    # Models have run, lets plot the stats
    all_stats = []
    labels = []
    for config in configs:
        all_stats.append(load_stats(config['save_stats']))
        labels.append(config['name'])

    # For every config, plot the loss across number of epochs
    plt = compare_training_stats(all_stats, labels)
    save_plt(plt, 'loss_test_dropout')
    # plt.show WILL WIPE THE PLT, so make sure you save the plot before you show it
    plt.show()

    # For every config, plot the accuracy across the number of epochs
    plt = compare_training_stats(all_stats, labels, metric_to_compare='accuracy', y_label='accuracy', title='Accuracy vs Epoch')
    save_plt(plt, 'acc_test_dropout')
    # plt.show WILL WIPE THE PLT, so make sure you save the plot before you show it
    plt.show()

    for data in test_data:
        print(f'{data["name"]}: {data["accuracy"]}')


    ## Just training for now ##

    # load the model from the saved file
    #net = load_modal('dropout_model')
    # switch model to eval mode so dropout does not get applied
    #net.eval()

    # Test the model on the test dataloader
    #test(net, test_dataloader)

"""
 No dropout, LR: 0.001: 44.2
 No dropout, LR: 0.01: 1.0
 
 Dropoiut of 25%, LR: 0.001: 45.03
 Dropoiut of 25%, LR: 0.01: 1.0
 
 Dropout of 50%, LR: 0.001: 38.02
 Dropout of 50%, LR: 0.01: 1.0
 Dropout of 50%, LR: 0.003: 27.3
 Dropout of 50%, LR: 0.007: 1.0
 
 Dropout of 75%, LR: 0.001: 22.07
 Dropout of 75%, LR: 0.01: 1.08
"""