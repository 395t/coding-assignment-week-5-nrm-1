from src.paths import DATA_DIR
from src.viz_helper import compare_training_stats, save_plt
from src.lifecycles import load_stats


def create_STL10_003_charts():
    training_loss_stat_files_003 = [
        "STL10_baseline_train_metrics",
        "WN_bb_my_norm_training@0.003|Adam|STL10",
        "STL10_batch_norm_3x_train_metrics",
        "DO_STL10|net_dropout_50_003",
        "STL10_ln_003_train_metrics"
    ]

    bl = load_stats(training_loss_stat_files_003[0])
    wn = load_stats(training_loss_stat_files_003[1])
    bn = load_stats(training_loss_stat_files_003[2])
    do = load_stats(training_loss_stat_files_003[3])
    ln = load_stats(training_loss_stat_files_003[4])

    labels = [
        "Base Line (no norm)",
        "Weight Normalization",
        "Batch Normalization",
        "Drop Out 50%",
        "Layer Normalization"
    ]

    training_sets = [
        bl,
        wn,
        bn,
        do,
        ln
    ]

    # For every config, plot the accuracy across the number of epochs
    plt = compare_training_stats(training_sets, labels, metric_to_compare='loss', y_label='training loss',
                                 title='LR 0.003: Loss vs Epoch', legend_loc='lower right')
    save_plt(plt, "STL10-Loss-vs-Epoch-003-Training")
    # plt.show WILL WIPE THE PLT, so make sure you save the plot before you show it
    plt.show()


def create_STL10_custom_charts():
    training_loss_stat_files_003 = [
        "STL10_baseline_train_metrics",
        "WN_bb_my_norm_training@0.001|Adam|STL10",
        "STL10_batch_norm_5x_train_metrics",
        "DO_STL10|net_dropout_50_lr1",
        "STL10_ln_001_train_metrics"
    ]

    bl = load_stats(training_loss_stat_files_003[0])
    wn = load_stats(training_loss_stat_files_003[1])
    bn = load_stats(training_loss_stat_files_003[2])
    do = load_stats(training_loss_stat_files_003[3])
    ln = load_stats(training_loss_stat_files_003[4])

    training_sets = [
        bl,
        wn,
        bn,
        do,
        ln,
    ]

    labels = [
        "Base Line (no norm)",
        "Weight Normalization @ 0.001",
        "Batch Normalization @ 0.005",
        "Drop Out 50% @ 0.001",
        "Layer Normalization @ 0.001"
    ]

    # For every config, plot the accuracy across the number of epochs
    plt = compare_training_stats(training_sets, labels, metric_to_compare='loss', y_label='training loss',
                                 title='Custom LR: Loss vs Epoch', legend_loc='lower right')
    save_plt(plt, "STL10-Loss-vs-Epoch-custom-Training")
    # plt.show WILL WIPE THE PLT, so make sure you save the plot before you show it
    plt.show()


def create_TINY_003_charts():
    training_loss_stat_files_003 = [
        "TINY_baseline_train_metrics",
        "WN_bb_my_norm_training@0.003|Adam|TINY",
        "TINY_batch_norm_3x_train_metrics",
        "DO_TINY|net_dropout_50_003",
        "TINY_ln_003_train_metrics"
    ]

    bl = load_stats(training_loss_stat_files_003[0])
    wn = load_stats(training_loss_stat_files_003[1])
    bn = load_stats(training_loss_stat_files_003[2])
    do = load_stats(training_loss_stat_files_003[3])
    ln = load_stats(training_loss_stat_files_003[4])

    labels = [
        "Base Line (no norm)",
        "Weight Normalization",
        "Batch Normalization",
        "Drop Out 50%",
        "Layer Normalization"
    ]

    training_sets = [
        bl,
        wn,
        bn,
        do,
        ln
    ]

    # For every config, plot the accuracy across the number of epochs
    plt = compare_training_stats(training_sets, labels, metric_to_compare='loss', y_label='training loss',
                                 title='lr 0.003: Loss vs Epoch', legend_loc='lower right')
    save_plt(plt, "TINY-Loss-vs-Epoch-003-Training")
    # plt.show WILL WIPE THE PLT, so make sure you save the plot before you show it
    plt.show()


def create_TINY_custom_charts():
    training_loss_stat_files_003 = [
        "TINY_baseline_train_metrics",
        "WN_bb_my_norm_training@0.003|Adam|TINY",
        "TINY_batch_norm_5x_train_metrics",
        "DO_TINY|net_dropout_50_lr1",
        "TINY_ln_001_train_metrics"
    ]

    bl = load_stats(training_loss_stat_files_003[0])
    wn = load_stats(training_loss_stat_files_003[1])
    bn = load_stats(training_loss_stat_files_003[2])
    do = load_stats(training_loss_stat_files_003[3])
    ln = load_stats(training_loss_stat_files_003[4])

    training_sets = [
        bl,
        wn,
        bn,
        do,
        ln,
    ]

    labels = [
        "Base Line (no norm)",
        "Weight Normalization @ 0.001",
        "Batch Normalization @ 0.005",
        "Drop Out 50% @ 0.001",
        "Layer Normalization @ 0.001"
    ]

    # For every config, plot the accuracy across the number of epochs
    plt = compare_training_stats(training_sets, labels, metric_to_compare='loss', y_label='training loss',
                                 title='Custom LR: Loss vs Epoch', legend_loc='lower right')
    save_plt(plt, "TINY-Loss-vs-Epoch-custom-Training")
    # plt.show WILL WIPE THE PLT, so make sure you save the plot before you show it
    plt.show()



if __name__ == "__main__":
    create_STL10_custom_charts()
    create_STL10_003_charts()
    create_TINY_custom_charts()
    create_TINY_003_charts()