from src.paths import DATA_DIR
from src.viz_helper import compare_training_stats, save_plt
from src.lifecycles import load_stats


def create_STL10_003_charts():
    training_loss_stat_files_003 = [
        "STL10_baseline_train_metrics",
        "WN_bb_my_norm_training@0.003|Adam|STL10",
        "STL10_batch_norm_3x_train_metrics",
        "DO_STL10|net_dropout_50_003",
        "STL10_ln_003_train_metrics",
        "STL10_insnorm_3x_train_metrics"
    ]

    bl = load_stats(training_loss_stat_files_003[0])
    wn = load_stats(training_loss_stat_files_003[1])
    bn = load_stats(training_loss_stat_files_003[2])
    do = load_stats(training_loss_stat_files_003[3])
    ln = load_stats(training_loss_stat_files_003[4])
    _in = load_stats(training_loss_stat_files_003[5])

    labels = [
        "Base Line (no norm)",
        "Weight Normalization",
        "Batch Normalization",
        "Drop Out 50%",
        "Layer Normalization",
        "Instance Normalization"
    ]

    training_sets = [
        bl,
        wn,
        bn,
        do,
        ln,
        _in
    ]

    # For every config, plot the accuracy across the number of epochs
    plt = compare_training_stats(training_sets, labels, metric_to_compare='loss', y_label='training loss',
                                 title='LR 0.003: Loss vs Epoch', legend_loc='upper right')
    save_plt(plt, "STL10-Loss-vs-Epoch-003-Training")
    # plt.show WILL WIPE THE PLT, so make sure you save the plot before you show it
    plt.show()


def create_STL10_custom_charts():
    training_loss_stat_files_003 = [
        "STL10_baseline_train_metrics",
        "WN_bb_my_norm_training@0.001|Adam|STL10",
        "STL10_batch_norm_5x_train_metrics",
        "DO_STL10|net_dropout_50_lr1",
        "STL10_ln_001_train_metrics",
        "STL10_insnorm_train_metrics"
    ]

    bl = load_stats(training_loss_stat_files_003[0])
    wn = load_stats(training_loss_stat_files_003[1])
    bn = load_stats(training_loss_stat_files_003[2])
    do = load_stats(training_loss_stat_files_003[3])
    ln = load_stats(training_loss_stat_files_003[4])
    _in = load_stats(training_loss_stat_files_003[5])

    training_sets = [
        bl,
        wn,
        bn,
        do,
        ln,
        _in,
    ]

    labels = [
        "Base Line (no norm)",
        "Weight Normalization @ 0.001",
        "Batch Normalization @ 0.005",
        "Drop Out 50% @ 0.001",
        "Layer Normalization @ 0.001",
        "Instance Normalization @ 0.001"
    ]

    # For every config, plot the accuracy across the number of epochs
    plt = compare_training_stats(training_sets, labels, metric_to_compare='loss', y_label='training loss',
                                 title='Custom LR: Loss vs Epoch', legend_loc='upper right')
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
                                 title='lr 0.003: Loss vs Epoch', legend_loc='upper right')
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
                                 title='Custom LR: Loss vs Epoch', legend_loc='upper right')
    save_plt(plt, "TINY-Loss-vs-Epoch-custom-Training")
    # plt.show WILL WIPE THE PLT, so make sure you save the plot before you show it
    plt.show()


def create_CIFAR_003_charts():
    training_loss_stat_files_003 = [
        "WN_cpc_no_norm_training@0.001|Adam|CIFAR-100",
        "WN_bb_my_norm_training@0.003|Adam|CIFAR-100",
        "CIFAR-100_batch_norm_3x_train_metrics",
        "net_dropout_50_003",
        "CIFAR-100_ln_003_train_metrics",
        "CIFAR-100_insnorm_3x_train_metrics"
    ]

    tmp_bl = load_stats(training_loss_stat_files_003[0])  # base line went longer than 20 epochs
    bl = {}

    for (ky, itm) in list(tmp_bl.items())[0:20]:
        bl[ky] = itm

    wn = load_stats(training_loss_stat_files_003[1])
    bn = load_stats(training_loss_stat_files_003[2])
    do = load_stats(training_loss_stat_files_003[3])
    ln = load_stats(training_loss_stat_files_003[4])
    _in = load_stats(training_loss_stat_files_003[5])

    labels = [
        "Base Line (no norm)",
        "Weight Normalization",
        "Batch Normalization",
        "Drop Out 50%",
        "Layer Normalization",
        "Instance Normalization"
    ]

    training_sets = [
        bl,
        wn,
        bn,
        do,
        ln,
        _in
    ]

    # For every config, plot the accuracy across the number of epochs
    plt = compare_training_stats(training_sets, labels, metric_to_compare='loss', y_label='training loss',
                                 title='lr 0.003: Loss vs Epoch', legend_loc='upper right')
    save_plt(plt, "CIFAR100-Loss-vs-Epoch-003-Training")
    # plt.show WILL WIPE THE PLT, so make sure you save the plot before you show it
    plt.show()


def create_CIFAR_custom_charts():
    training_loss_stat_files_003 = [
        "WN_cpc_no_norm_training@0.001|Adam|CIFAR-100",
        "WN_bb_my_norm_training@0.001|Adam|CIFAR-100",
        "CIFAR-100_batch_norm_5x_train_metrics",
        "net_dropout_50_lr1",
        "CIFAR-100_ln_001_train_metrics",
        "CIFAR-100_insnorm_7x_train_metrics"
    ]

    tmp_bl = load_stats(training_loss_stat_files_003[0])  # base line went longer than 20 epochs
    bl = {}

    for (ky, itm) in list(tmp_bl.items())[0:20]:
        bl[ky] = itm

    wn = load_stats(training_loss_stat_files_003[1])
    bn = load_stats(training_loss_stat_files_003[2])
    do = load_stats(training_loss_stat_files_003[3])
    ln = load_stats(training_loss_stat_files_003[4])
    _in = load_stats(training_loss_stat_files_003[5])


    labels = [
        "Base Line (no norm)",
        "Weight Normalization @ 0.001",
        "Batch Normalization @ 0.005",
        "Drop Out 50% @ 0.001",
        "Layer Normalization @ 0.001",
        "Instance Normalization @ 0.007"
    ]

    training_sets = [
        bl,
        wn,
        bn,
        do,
        ln,
        _in
    ]

    # For every config, plot the accuracy across the number of epochs
    plt = compare_training_stats(training_sets, labels, metric_to_compare='loss', y_label='training loss',
                                 title='lr Custom: Loss vs Epoch', legend_loc='upper right')
    save_plt(plt, "CIFAR100-Loss-vs-Epoch-custom-Training")
    # plt.show WILL WIPE THE PLT, so make sure you save the plot before you show it
    plt.show()


def create_ViT_CIFAR_charts():
    training_loss_stat_files_003 = [
        "ViT_LayerNorm_CIFAR-100_training_stats",
        "ViT_WeightNorm_CIFAR-100_training_stats",
        "ViT_BatchNorm_CIFAR-100_training_stats",
        "ViT_DropOut_CIFAR-100_training_stats",
        "ViT_InstanceNorm_CIFAR-100_training_stats"
    ]

    ln = load_stats(training_loss_stat_files_003[0])
    wn = load_stats(training_loss_stat_files_003[1])
    bn = load_stats(training_loss_stat_files_003[2])
    do = load_stats(training_loss_stat_files_003[3])
    _in = load_stats(training_loss_stat_files_003[4])

    labels = [
        "Layer Normalization",
        "Weight Normalization",
        "Batch Normalization",
        "Drop Out 50%",
        "Instance Norm",
    ]

    training_sets = [
        ln,
        wn,
        bn,
        do,
       _in,
    ]

    # For every config, plot the accuracy across the number of epochs
    plt = compare_training_stats(training_sets, labels, metric_to_compare='loss', y_label='training loss',
                                 title='CIFAR100 ViT Loss vs Epoch', legend_loc='upper right')
    save_plt(plt, "ViT_CIFAR100-Loss-vs-Epoch-custom-Training")
    # plt.show WILL WIPE THE PLT, so make sure you save the plot before you show it
    plt.show()

def create_ViT_STL10_charts():
    training_loss_stat_files_003 = [
        "ViT_LayerNorm_STL10_training_stats",
        "ViT_WeightNorm_STL10_training_stats",
        "ViT_BatchNorm_STL10_training_stats",
        "ViT_DropOut_STL10_training_stats",
        "ViT_InstanceNorm_STL10_training_stats"
    ]

    ln = load_stats(training_loss_stat_files_003[0])
    wn = load_stats(training_loss_stat_files_003[1])
    bn = load_stats(training_loss_stat_files_003[2])
    do = load_stats(training_loss_stat_files_003[3])
    _in = load_stats(training_loss_stat_files_003[4])

    labels = [
        "Layer Normalization",
        "Weight Normalization",
        "Batch Normalization",
        "Drop Out 50%",
        "Instance Norm",
    ]

    training_sets = [
        ln,
        wn,
        bn,
        do,
       _in,
    ]

    # For every config, plot the accuracy across the number of epochs
    plt = compare_training_stats(training_sets, labels, metric_to_compare='loss', y_label='training loss',
                                 title='STL10 ViT Loss vs Epoch', legend_loc='upper right')
    save_plt(plt, "ViT_STL10-Loss-vs-Epoch-custom-Training")
    # plt.show WILL WIPE THE PLT, so make sure you save the plot before you show it
    plt.show()


def create_ViT_TINY_charts():
    training_loss_stat_files_003 = [
        "ViT_LayerNorm_TINY_training_stats",
        "ViT_WeightNorm_TINY_training_stats",
        "ViT_BatchNorm_TINY_training_stats",
        "ViT_DropOut_TINY_training_stats",
        "ViT_InstanceNorm_TINY_training_stats"
    ]

    ln = load_stats(training_loss_stat_files_003[0])
    wn = load_stats(training_loss_stat_files_003[1])
    bn = load_stats(training_loss_stat_files_003[2])
    do = load_stats(training_loss_stat_files_003[3])
    _in = load_stats(training_loss_stat_files_003[4])

    labels = [
        "Layer Normalization",
        "Weight Normalization",
        "Batch Normalization",
        "Drop Out 50%",
        "Instance Norm",
    ]

    training_sets = [
        ln,
        wn,
        bn,
        do,
        _in,
    ]

    # For every config, plot the accuracy across the number of epochs
    plt = compare_training_stats(training_sets, labels, metric_to_compare='loss', y_label='training loss',
                                 title='TINY ViT Loss vs Epoch', legend_loc='upper right')
    save_plt(plt, "ViT_TINY-Loss-vs-Epoch-custom-Training")
    # plt.show WILL WIPE THE PLT, so make sure you save the plot before you show it
    plt.show()


if __name__ == "__main__":
    create_STL10_custom_charts()
    create_STL10_003_charts()
    create_TINY_custom_charts()
    create_TINY_003_charts()
    create_CIFAR_custom_charts()
    create_CIFAR_003_charts()

    create_ViT_CIFAR_charts()
    create_ViT_STL10_charts()
    create_ViT_TINY_charts()