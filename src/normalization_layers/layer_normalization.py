
import src
import pickle as pkl
import torch
from torch import nn
from functools import partial

from src.paths import CHECKPOINTS_DIR, DATA_DIR
from src.lifecycles import train, test, test_validation, save_model, load_modal, save_stats, load_stats
from src.viz_helper import compare_training_stats, save_plt
from src.backbone import BackboneLayerNorm

if __name__ == "__main__":
    for dataset, num_classes, img_size, epoch in (('TINY', 200, 64, 10),("STL10", 10, 96, 20),("CIFAR-100", 100, 32, 20)):
        # Experiments to reproduce from batch norm paper
        experiments = {
            'ln_001': {'mod': BackboneLayerNorm(num_classes=num_classes, img_size=img_size), 'lr': 0.001},
            'ln_003': {'mod': BackboneLayerNorm(num_classes=num_classes, img_size=img_size), 'lr': 0.003},
        }
        # Grab the dataset, with a batch size of 10, and store it in the Data Directory (src/data)
        train_dataloader, test_dataloader = src.get_dataloder(dataset, 64, DATA_DIR)

        for exp_name, config in experiments.items():
            print(f'---Running experiment {exp_name} on dataset {dataset} ---')

            # Create the backbone network with 100 classes and batch norm
            net: nn.Module = config['mod']

            # Set up a learning rate and optimizer
            LR = config['lr']
            optimizer = torch.optim.Adam(net.parameters(), lr=LR)

            # Train the network on the Adam optimizer, using the training data loader, for 10 epochs
            print('---Training---')

            # can use below for quick test
            # train_metrics = train(net, optimizer, list(train_dataloader)[:1], epochs=2)
            train_metrics = train(net, optimizer, train_dataloader, epochs=epoch)

            save_path = f'{dataset}_{exp_name}_train_metrics'
            save_stats(train_metrics, save_path)

            # Save the model for use later in the checkpoints directory (src/checkpoints) as 'example_model.pt'
            save_model(net, exp_name)

            # load the model from the saved file
            net = load_modal(exp_name)

            # Test the model on the test dataloader
            print('---Validation---')

            # can use below for quick test
            # val_metrics = test_validation(net, list(test_dataloader)[:1])
            val_metrics = test_validation(net, test_dataloader)

            save_path = f'{dataset}_{exp_name}_val_metrics'
            save_stats(val_metrics, save_path)


        # Models have run, lets plot the stats
        all_stats = []
        labels = []
        for exp_name in experiments.keys():
            load_path = f'{dataset}_{exp_name}_train_metrics'
            all_stats.append(load_stats(load_path))
            labels.append(exp_name)

        # For every config, plot the loss across number of epochs
        plt = compare_training_stats(all_stats, labels)
        save_plt(plt, f'{dataset}_loss_train')
        # plt.show WILL WIPE THE PLT, so make sure you save the plot before you show it
        plt.show()

        # aggregate test accuracies across experiments and save
        test_acc = {}
        for exp_name in experiments.keys():
            load_path = f'{dataset}_{exp_name}_val_metrics'
            stats = load_stats(load_path)
            exp = f'{dataset}_{exp_name}'
            test_acc[exp] = stats['accuracy']
        save_stats(test_acc, f'{dataset}_test_acc')