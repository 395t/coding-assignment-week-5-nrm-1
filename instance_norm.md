# Instance Normalization Empirical Analysis

We analyze the performance of instance normalization with the same backbone network on three datasets: CIFAR-100, STL10, TinyImagenet.
Per our evaluation protocol, we run all the experiments on several learning rates ranging from 0.001 to 0.03 and we show their respective training curves(loss vs. epoch) and their best-performing test accuracy.
The metric used for test is top-1 accuracy and we train for 16 epoches for each experiment.

Note that there's a difference in training for this set of models: we used a much larger batch size in order to speed up.
Therefore, the baseline accuracy was inflated compared to other set of results, which, shows another aspect of normalization that
normalizations tend to get more accurate gradient estimations even batch size is small.

# Quantitative Results
| |**CIFAR-100 Test Accuracy**|**STL-10 Test Accuracy**|**TinyImageNet Test Accuracy**|
|:-----:|:-----:|:-----:|:-----:|
|baseline|61.27|55.9|*|
|instance_norm(1xlr)|61.75|59.52|*|
|instance_norm(3xlr)|54.53|46.98|*|
|instance_norm(7xlr)|47.07|10|*|

# Training Loss plot
