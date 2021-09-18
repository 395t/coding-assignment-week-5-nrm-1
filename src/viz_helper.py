from src.paths import IMAGES_DIR

from typing import List
import matplotlib.pyplot as plt
from pathlib import Path


def compare_training_stats(
        training_stats: List[dict],
        labels: List[str],
        metric_to_compare: str = "loss",
        x_label: str = 'epoch',
        y_label='loss',
        legend_loc='upper right',
        title='Loss vs Epoch'
    ):

    metric_values = []

    for training_stat in training_stats:
        current_metric_values = []
        for stats in training_stat.values():
            current_metric_values.append(stats[metric_to_compare])
        metric_values.append(current_metric_values)


    for values, label in zip(metric_values, labels):
        plt.plot(values, '-x', label=label)

    plt.xlabel(x_label)
    plt.ylabel(y_label)
    plt.legend(loc=legend_loc)
    plt.title(title)

    return plt


def save_plt(plot, filename: str, location: Path = IMAGES_DIR):
    plot.draw()
    plot.savefig(f'{str(location / filename)}.png')