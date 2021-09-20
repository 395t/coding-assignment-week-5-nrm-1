from src.paths import DATA_DIR
from src.viz_helper import compare_training_stats, save_plt
from src.lifecycles import load_stats


def create_STL10_003_charts():
    training_loss_stat_files_003 = [
        "STL10_baseline_train_metrics",
        "WN_bb_my_norm_training@0.003|Adam|STL10",
        "STL10_batch_norm_3x_train_metrics",
        "DO_STL10|net_dropout_50_003"
    ]

    bl = load_stats(training_loss_stat_files_003[0])
    wn = load_stats(training_loss_stat_files_003[1])
    bn = load_stats(training_loss_stat_files_003[2])
    do = load_stats(training_loss_stat_files_003[3])

    labels = [
        "Base Line (no norm)",
        "Weight Normalization",
        "Batch Normalization",
        "Drop Out 50%",
    ]

    # For every config, plot the accuracy across the number of epochs
    plt = compare_training_stats([bl, wn, bn, do], labels, metric_to_compare='loss', y_label='training loss',
                                 title='Loss vs Epoch', legend_loc='lower right')
    save_plt(plt, "STL10-Loss-vs-Epoch-003-Training")
    # plt.show WILL WIPE THE PLT, so make sure you save the plot before you show it
    plt.show()


def create_STL10_custom_charts():
    training_loss_stat_files_003 = [
        "STL10_baseline_train_metrics",
        "WN_bb_my_norm_training@0.001|Adam|STL10",
        "STL10_batch_norm_5x_train_metrics",
        # "DO_STL10|net_dropout_50_001"
    ]

    bl = load_stats(training_loss_stat_files_003[0])
    wn = load_stats(training_loss_stat_files_003[1])
    bn = load_stats(training_loss_stat_files_003[2])
    # do = load_stats(training_loss_stat_files_003[3])

    training_sets = [
        bl,
        wn,
        bn,
        #do,
    ]

    labels = [
        "Base Line (no norm)",
        "Weight Normalization @ 0.001",
        "Batch Normalization @ 0.005",
        # "Drop Out 50% @ 0.001",
    ]

    # For every config, plot the accuracy across the number of epochs
    plt = compare_training_stats(training_sets, labels, metric_to_compare='loss', y_label='training loss',
                                 title='Loss vs Epoch', legend_loc='lower right')
    save_plt(plt, "STL10-Loss-vs-Epoch-custom-Training")
    # plt.show WILL WIPE THE PLT, so make sure you save the plot before you show it
    plt.show()


if __name__ == "__main__":
    create_STL10_custom_charts()
    create_STL10_003_charts()