import src
import torch
from torch import nn
import torch.nn.functional as F
from typing import Tuple

from functools import partial

from src.paths import CHECKPOINTS_DIR, DATA_DIR
from src.lifecycles import train, test, save_model, load_modal, save_stats, load_stats
from src.viz_helper import compare_training_stats, save_plt


# Custom Implementation of Weight Normalization
class WeightNorm(nn.Module):
    def __init__(self, layer: nn.Module, weight_name: str = "weight", dim: int = 0, divide_w_by_g_in_init:bool = True):
        super(WeightNorm, self).__init__()

        self.weight_name = weight_name
        self.dim = dim

        self.layer: nn.Module = layer

        self._setup(divide_w_by_g_in_init)

    def _setup(self, divide_w_by_g_in_init:bool = True):
        # Fetch the weights for the current layer passed in
        # This weight is what we are re-parameterizing
        # In order to do this correctly we have to remove the weight matrix from pytorches internal table of weights
        # that can be back propped against, and add our new G and V weights instead

        # Modules weights we are going to reparam.
        weight = getattr(self.layer, self.weight_name)

        # Remove the weights from the models parameter list.  This will mean it won't calc backprop on it later
        if self.weight_name in self.layer._parameters:
            del self.layer._parameters[self.weight_name]


        # Initialize g
        # From the paper we have
        # g = ||w||

        # had to look this up, norm_except_dim function allows you to normalize all dimensions of a tensor except
        # the one you specify.  So although torch.linalg.norm will work for dense layers (matrices) it does not
        # work for conv layers.
        # g = torch.linalg.norm(weight)
        g = torch.norm_except_dim(weight, 2, dim=self.dim)

        # Initialize v
        # From the paper we have
        # w/g = v / ||v||

        # Really, we are not calculating v here.  Instead we are calculating the value v / ||v|| for the initialization
        # although when we call forward, we will use v / ||v||

        if divide_w_by_g_in_init:
            v = weight / g
        else:
            # Pytorch's implementation does not divide by g (so this version is closer to what people use in production)
            # I think this is because g is just a scalar and can be adjusted for later in the training process.
            # However, its a cool experiment to see which does better.
            v = weight

        # We need to register the new parameters g and v to our model.  Following some best practices we will use
        # the same name as the weight we are re-parameterizing but with a suffix of _g or _v according to the param
        self.layer.register_parameter(f'{self.weight_name}_g', nn.Parameter(g.data))  # g.data to avoid references to w
        self.layer.register_parameter(f'{self.weight_name}_v', nn.Parameter(v.data))

        # Update the layer params.  This probably is unnecessary since every forward call will have this called anyway
        self.set_W()


    @staticmethod
    def get_weight(g: torch.Tensor, v: torch.Tensor, dim: int):
        return (g / v.norm(dim=dim, keepdim=True)) * v

    @property
    def g(self):
        # quick helper for getting the g vector -> self.g == g vector
        return getattr(self.layer, f'{self.weight_name}_g')

    @property
    def v(self):
        # quick helper for getting the v matrix -> self.v == v matrix
        return getattr(self.layer, f'{self.weight_name}_v')

    def set_W(self):
        # This function will reparam the weight matrix of the currently stored layer.

        W = self.get_weight(self.g, self.v, self.dim)
        # For the layer we are going to pass input into, we want to reset the weight matrix we are normalizing to our
        # new parameterization
        setattr(self.layer, self.weight_name, W)

    def forward(self, x):
        # For every forward call, we need to reparam the weight matrix.
        self.set_W()
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


def test_weight_norm(
        epochs: int = 40,
        batch_size: int = 64,
        learning_rate: float = 0.001,
        optimizer: torch.optim.Optimizer = torch.optim.Adam,

        use_bb: bool = True,
        use_cnc: bool = True,
        include_my_weight_norm: bool = True,
        include_no_weight_norm: bool = True,
        include_pytorch_weight_norm: bool = True,
        include_no_divide_by_g: bool = False,

        train_models: bool = True,
        test_models: bool = True,

        loss_fig_title: str = None,
        acc_fig_title: str = None,

        test_loss_fig_title: str = None,
        test_acc_fig_title: str = None,

        dataset: str = "CIFAR-100",

        checkpoint: int = -1,
        save_on_checkpoint: bool = True,

    ):

    EPOCHS = epochs
    BATCH_SIZE = batch_size
    optim_name = optimizer.__name__

    if loss_fig_title is None:
        loss_fig_title = f'WNT_training_loss'
    if acc_fig_title is None:
        acc_fig_title = f'WNT_training_acc'
    if test_loss_fig_title is None:
        test_loss_fig_title = f'WNT_test_loss'
    if test_acc_fig_title is None:
        test_acc_fig_title = f'WNT_test_acc'

    if checkpoint == -1:
        checkpoint = EPOCHS

    loss_fig_title += f'_{learning_rate}|{optim_name}|{dataset}'
    acc_fig_title += f'_{learning_rate}|{optim_name}{dataset}'
    test_loss_fig_title += f'_{learning_rate}|{optim_name}|{dataset}'
    test_acc_fig_title += f'_{learning_rate}|{optim_name}|{dataset}'

    NUM_CLASSES = 100
    if dataset == 'STL10':
        NUM_CLASSES = 10

    my_norm_w_g_norm = partial(WeightNorm, divide_w_by_g_in_init=True)
    my_norm_wo_g_norm = partial(WeightNorm, divide_w_by_g_in_init=False)


    net_cpc_torch_norm = ConvPoolCNNC(normalizer=nn.utils.weight_norm, num_classes=NUM_CLASSES)
    net_cpc_my_norm = ConvPoolCNNC(normalizer=my_norm_w_g_norm, num_classes=NUM_CLASSES)
    net_cpc_my_norm_wo_g_norm = ConvPoolCNNC(normalizer=my_norm_wo_g_norm, num_classes=NUM_CLASSES)
    net_cpc_no_norm = ConvPoolCNNC(normalizer=nn.Sequential, num_classes=NUM_CLASSES)

    # I copied the backbone and modified it slightly to work for my paper (I have to have access to the layer,
    # others may not need this though).
    net_bb_torch_norm = WNBackbone(NUM_CLASSES, norm_mod=nn.utils.weight_norm)
    net_bb_my_norm = WNBackbone(NUM_CLASSES, norm_mod=my_norm_w_g_norm)
    net_bb_my_norm_wo_g_norm = WNBackbone(NUM_CLASSES, norm_mod=my_norm_wo_g_norm)
    net_bb_no_norm = WNBackbone(NUM_CLASSES, norm_mod=nn.Sequential)

    # Different models with some parameters I want to compare against
    configs = []

    if use_bb:
        if include_pytorch_weight_norm:
            configs.append({'name': f'Backbone with Torch Weight Norm LR {learning_rate} USING {optim_name} ON {dataset}', 'label': 'BackBone Torch', 'model': net_bb_torch_norm, 'save_model': f'WN_bb_torch_norm@{learning_rate}|{optim_name}{dataset}', 'save_stats': f'WN_bb_torch_norm_training@{learning_rate}|{optim_name}|{dataset}', 'LR': learning_rate})

        if include_no_weight_norm:
            configs.append({'name': f'Backbone with No Weight Norm LR {learning_rate} USING {optim_name} ON {dataset}', 'label':'BackBone None', 'model': net_bb_no_norm, 'save_model': f'WN_bb_no_norm@{learning_rate}|{optim_name}|{dataset}', 'save_stats': f'WN_bb_no_norm_training@{learning_rate}|{optim_name}|{dataset}', 'LR': learning_rate})

        if include_my_weight_norm:
            configs.append({'name': f'Backbone with My Weight Norm LR {learning_rate} USING {optim_name} ON {dataset}', 'label': 'BackBone Mine', 'model': net_bb_my_norm, 'save_model': f'WN_bb_my_norm@{learning_rate}|{optim_name}|{dataset}', 'save_stats': f'WN_bb_my_norm_training@{learning_rate}|{optim_name}|{dataset}', 'LR': learning_rate})

        if include_no_divide_by_g:
            configs.append({'name': f'Backbone with My Weight NGN Norm LR {learning_rate} USING {optim_name} ON {dataset}', 'label': 'BackBone Mine NGN', 'model': net_bb_my_norm_wo_g_norm, 'save_model': f'WN_bb_my_norm_no_gn@{learning_rate}|{optim_name}|{dataset}', 'save_stats': f'WN_bb_my_norm_no_gn_training@{learning_rate}|{optim_name}|{dataset}', 'LR': learning_rate})


    if use_cnc:
        if include_pytorch_weight_norm:
            configs.append({'name': f'Conv Pool C with Torch Weight Norm LR {learning_rate} USING {optim_name} ON {dataset}', 'label': 'CPC Torch', 'model': net_cpc_torch_norm, 'save_model': f'WN_cpc_torch_norm@{learning_rate}|{optim_name}|{dataset}', 'save_stats': f'WN_cpc_torch_norm_training@{learning_rate}|{optim_name}|{dataset}', 'LR': learning_rate})

        if include_no_weight_norm:
            configs.append({'name': f'Conv Pool C with No Weight Norm LR {learning_rate} USING {optim_name} ON {dataset}', 'label': 'CPC None', 'model': net_cpc_no_norm, 'save_model': f'WN_cpc_no_norm@{learning_rate}|{optim_name}|{dataset}', 'save_stats': f'WN_cpc_no_norm_training@{learning_rate}|{optim_name}|{dataset}', 'LR': learning_rate})

        if include_my_weight_norm:
            configs.append({'name': f'Conv Pool C with My Weight Norm LR {learning_rate} USING {optim_name} ON {dataset}', 'label': 'CPC Mine', 'model': net_cpc_my_norm, 'save_model': f'WN_cpc_my_norm@{learning_rate}|{dataset}', 'save_stats': f'WN_cpc_my_norm_training@{learning_rate}|{optim_name}|{dataset}', 'LR': learning_rate})

        if include_no_divide_by_g:
            configs.append({'name': f'Conv Pool C with My Weight Norm NGN LR {learning_rate} USING {optim_name} ON {dataset}', 'label': 'CPC Mine NGN', 'model': net_cpc_my_norm_wo_g_norm, 'save_model': f'WN_cpc_my_norm_no_gn@{learning_rate}|{dataset}', 'save_stats': f'WN_cpc_my_norm_no_gn_training@{learning_rate}|{optim_name}|{dataset}', 'LR': learning_rate})


    # Train each model
    if train_models:
        for config in configs:
            test_stats = {}
            train_stats = {}

            net = config['model']

            # Grab the CIFAR-100 dataset, with a batch size of 10, and store it in the Data Directory (src/data)
            train_dataloader, test_dataloader = src.get_dataloder(dataset, BATCH_SIZE, DATA_DIR)

            # Set up a learning rate and optimizer
            opt = optimizer(net.parameters(), lr=config['LR'])

            for i in range(0, EPOCHS, checkpoint):
                # Train the network on the optimizer, using the training data loader, for EPOCHS epochs.
                stats = train(net, opt, train_dataloader, starting_epoch=i, epochs=i+checkpoint, loader_description=config['name'])
                train_stats.update(stats)

                if save_on_checkpoint:
                    # Save the model for testing later
                    save_model(net, f"EPOCH_{i}_checkpoint_{config['save_model']}")
                    # Save the stats from the training loop for later
                    save_stats(train_stats, f"EPOCH_{i}_checkpoint_{config['save_stats']}")

                if test_models:
                    test_stats[f'epoch_{i+1}'] = test(net, test_dataloader, loader_description=f'TESTING @ epoch {i}: {config["name"]}')
                    if save_on_checkpoint:
                        save_stats(train_stats, f"test_EPOCH_{i}_checkpoint_{config['save_stats']}")

            save_model(net, f"{config['save_model']}")
            # Save the stats from the training loop for later
            save_stats(train_stats, f"{config['save_stats']}")

            if test_models:
                test_stats[f'epoch_{EPOCHS + 1}'] = test(net, test_dataloader, loader_description=f'TESTING: {config["name"]}')

                save_stats(test_stats, f'test_{config["save_stats"]}')

    # Models have run, lets plot the stats
    train_stats = []
    test_stats = []
    labels = []
    for config in configs:
        train_stats.append(load_stats(config['save_stats']))
        test_stats.append(load_stats(f'test_{config["save_stats"]}'))
        labels.append(config['label'])


    if len(train_stats) and train_models:
        # For every config, plot the loss across number of epochs
        plt = compare_training_stats(train_stats, labels)
        save_plt(plt, loss_fig_title)
        # plt.show WILL WIPE THE PLT, so make sure you save the plot before you show it
        plt.show()

        # For every config, plot the accuracy across the number of epochs
        plt = compare_training_stats(train_stats, labels, metric_to_compare='accuracy', y_label='accuracy',
                                     title='Accuracy vs Epoch', legend_loc='lower right')
        save_plt(plt, acc_fig_title)
        # plt.show WILL WIPE THE PLT, so make sure you save the plot before you show it
        plt.show()


    if len(test_stats) > 0 and test_models:
        plt = compare_training_stats(test_stats, labels, title="Test Loss vs Checkpoint")
        save_plt(plt, test_loss_fig_title)
        plt.show()

        plt = compare_training_stats(test_stats, labels, metric_to_compare='accuracy', y_label='accuracy',
                                     title='Test Accuracy vs Checkpoint', legend_loc='lower right')
        save_plt(plt, test_acc_fig_title)
        plt.show()




if __name__ == "__main__":
    test_weight_norm(
        epochs=20,
        batch_size=64,
        learning_rate=0.001,
        optimizer=torch.optim.Adam,

        use_bb=True,
        use_cnc=True,
        include_my_weight_norm=True,
        include_no_weight_norm=False,
        include_pytorch_weight_norm=True,
        include_no_divide_by_g=True,

        train_models=True,
        test_models=True,

        loss_fig_title='WNT_training_loss_ngn_vs_gn',
        acc_fig_title='WNT_training_accuracy_ngn_vs_gn',
        test_loss_fig_title='WNT_testing_loss_ngn_vs_gn',
        test_acc_fig_title='WNT_testing_accuracy_ngn_vs_gn',

        dataset='CIFAR-100',
        checkpoint=1,
        save_on_checkpoint=False,
    )

