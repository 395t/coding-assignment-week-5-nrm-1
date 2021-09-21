import src
import pickle as pkl
import torch
from torch import nn
from functools import partial
from torch.cuda import amp

from src.paths import CHECKPOINTS_DIR, DATA_DIR
from src.lifecycles import train, test, test_validation, save_model,\
    load_modal, save_stats, load_stats, save_dict, load_dict
from src.viz_helper import compare_training_stats, save_plt


class IdentityNormLayer(nn.Module):
    def __init__(self, extra_parameter_saved_via_partial):
        super(IdentityNormLayer, self).__init__()

    def forward(self, x):
        return x


if __name__ == "__main__":
    dataset, num_classes = "STL10", 10
    experiments = {
        "baseline": {"norm": IdentityNormLayer, "lr": 1e-3},
        "insnorm": {"norm": nn.InstanceNorm2d, "lr": 1e-3},
        "insnorm_3x": {"norm": nn.InstanceNorm2d, "lr": 3e-3},
        "insnorm_30x": {"norm": nn.InstanceNorm2d, "lr": 3e-2},
    }

    train_loader, test_loader = src.get_dataloder(dataset, 128, "./data")

    for exp_name, config in experiments.items():
        print(f"---Exp {exp_name}, dataset {dataset}---")
        norm_mod = partial(config["norm"], 128)

        net = src.Backbone(num_classes, norm_mod)
        lr = config["lr"]
        opt = torch.optim.Adam(net.parameters(), lr=lr)
        scaler = amp.GradScaler()
        train_metrics = train(net, opt, train_loader, epochs=10, gradscaler=scaler, gpu="cuda:1")

        save_path = f"{dataset}_{exp_name}_train_metrics"
        save_stats(train_metrics, save_path)

        states = {"net": net.state_dict(), "opt": opt.state_dict(), "sclr": scaler.state_dict()}
        save_dict(states, exp_name)

        val_metrics = test_validation(net, test_loader)

        save_path = f"{dataset}_{exp_name}_val_metrics"
        save_stats(val_metrics, save_path)
