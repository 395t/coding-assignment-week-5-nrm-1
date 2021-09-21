Normalization Methods - Week 5 Group 1
-

Papers covered in our repo:

- Dropout: A Simple Way to Prevent Neural Networks from Overfitting, Srivastava, Hinton, Krizhevsky, Sutskever, Salakhutdinov; 2014	
- Batch Normalization: Accelerating Deep Network Training by Reducing Internal Covariate Shift, Ioffe, Szegedy; 2015	
- Weight Normalization: A Simple Reparameterization to Accelerate Training of Deep Neural Networks, Salimans, Kingma; 2016	
- Layer Normalization, Ba, Kiros, Hinton; 2016	
- Instance Normalization: The Missing Ingredient for Fast Stylization, Ulyanov, Vedaldi, Lempitsky; 2016	

### Datasets

- CIFAR 100
- STL10
- Tiny ImageNet

### Architectures


Our main architecture all normalizations were tested on was a ResNet backbone, however, we also
tested Weight Normalization on ConvPool-CNN-C (a model that was explored in the paper). The last
model we tried was a custom integration of the Visual Transformer ViT.

- Resnet
- ViT
- ConvPool-CNN-C

# Main Experiments 

### CIFAR 100 Task Performance per Normalization

Description:
    CIFAR 100 is a dataset with 100 classes, each image is a 32x32 image, and the goal is to classify 
    the images into their correct categories.

Models:
    We used CIFAR on all of our models because of its low resolution and smaller size, its easy to quickly
iterate on it.

Setup:
    For all models we followed these hyper-parameters

- Batch size of 64
- 20 epochs
- learning rate of 0.003 and one Custom LR run (paper owner chose a different LR)
    

![CIFAR100 Training Loss Chart lr 003](./images/CIFAR100-Loss-vs-Epoch-003-Training.png)
![CIFAR100 Training Loss Chart lr custom](./images/CIFAR100-Loss-vs-Epoch-custom-Training.png)

| Normalization | Test Acc @ 0.003 | Test Acc @ Custom |
| --- | --- | --- |
| Weight Normalization | 44.1 | 46.52 |
| Batch Normalization | 44.09 | 41.36 |
| Drop Out | 27.3 | 38.02 |
| Layer Normalization | 31.74 | 37.04 |
| Instance Normalization | 44.32 | 35.4 |

### STL10 Task Performance per Normalization

Description:
    STL10 has 10 classes with images of size 96x96.  You have a very small pool of training images, which
is really meant for training unsupervised algorithms but it also is helpful for testing generalization.

Models:
    STL10 was used on our Resnet Backbone for our experiments

Setup:
    For all models we followed these hyper-parameters

- Batch size of 64
- 20 epochs
- learning rate of 0.003

![STL10 Training Loss Chart lr 003](./images/STL10-Loss-vs-Epoch-003-Training.png)
![STL10 Training Loss Chart lr custom](./images/STL10-Loss-vs-Epoch-custom-Training.png)

| Normalization | Test Acc @ 0.003 | Test Acc @ Custom |
| --- | --- | --- |
| Weight Normalization | 43.3 | 46.6 |
| Batch Normalization | 56.8875 | 60.1 |
| Drop Out | 43.89 | 52.08 |
| Layer Normalization | 46.75 | 57.0 |
| Instance Normalization | 36.5125 |  52.125 | 

### Tiny ImageNet Task Performance per Normalization

Description:
    Tiny ImageNet is a small version of ImageNet, it has 100,000 images with 200 classes (500 images per class).
The images are of size 64 x 64.  For testing each image class has 50 test images.

Models:
    Tiny ImageNet was used on our Resnet Backbone for our experiments

Setup:
    For all models we followed these hyper-parameters

- Batch size of 64
- 10 epochs (it took too long to train for longer epochs)
- learning rate of 0.003 and one Custom LR run (paper owner chose a different LR)

![TINY Training Loss Chart lr 003](./images/TINY-Loss-vs-Epoch-003-Training.png)
![TINY Training Loss Chart lr custom](./images/TINY-Loss-vs-Epoch-custom-Training.png)

| Normalization | Test Acc @ 0.003 | Test Acc @ Custom |
| --- | --- | --- |
| Weight Normalization | 26.78 | 28.13 |
| Batch Normalization | 23.38 | 14.5 |
| Drop Out | 13.72 | 14.39 |
| Layer Normalization | 15.62 | 15.68 |

### ViT Tests

Description: To test a different architecture other than
Conv Nets, we ran our normalization layers on a ViT network
across all 3 datasets.

Setup:

- Batch size of 64
- 20 epochs (10 for Tiny ImageNet)
- Learning rate of 0.001



![ViT CIFAR100 Training Loss Chart lr 003](./images/ViT_CIFAR100-Loss-vs-Epoch-custom-Training.png)
![ViT STL10 Training Loss Chart lr custom](./images/ViT_STL10-Loss-vs-Epoch-custom-Training.png)
![ViT TINY Training Loss Chart lr custom](./images/ViT_TINY-Loss-vs-Epoch-custom-Training.png)

Test accuracies:

| Normalization | CIFAR 100 | STL10 | Tiny ImageNet |
| --- | --- | --- | --- |
| Weight Normalization | 5.02 | 10.6 | 0.79 |
| Batch Normalization | 27.5 | 42.18 | 14.65 |
| Drop Out | 26.07 | 43.29 | 12.96 |
| Layer Normalization | 28.34 | 42.14 | 13.71 |
| Instance Normalization | 30.32 | 43.6 | 13.25 |


# Other Experiments

## Dropout Experiments
One of the most used techniques to prevent overfitting on a neural network is dropout. The method consists of including a probability "p" of existing or not to each node of each layer on the network. Using a p value of 0.5 for dropout is usually the standard for networks.\
For this project, we tested dropout through a CNN implementation in Pytorch, training on CIFAR100, assigning dropout probabilities of 0 (no dropout), 0.25, 0.50, and 0.75, as well as two different learning rates: 0.01, 0.001.\
The dropout method was implemented through Pytorch's built in `nn.Dropout()` module, which takes in a single p (probability) parameter.

The model with no dropout and learning rate of 0.001 resulted in the highest training accuracy, at 14 epochs.\
The models with a learning rate = 0.01 resulted in really high loss results. They were taken off the result plots. Our understanding is that the LR was simply too large.\
Although 'no dropout' resulted in the best test accuracy, dropout of 50% was chosen for the comparison to other normalization methods, as it is the standard value used in most models.

### CIFAR100 Dropout Experiment
![Dropout Training Loss on CIFAR100](./images/DO_training_loss.png)
![Dropout Training Accuracy on CIFAR100](./images/DO_training_acc.png)

### STL10 and TinyImageNet Experiment
On the remaining two datasets, dropout was set to 0.5, and learning rate was tested at 0.001 (best performing), and 0.003 (normalizations comparison). The best performing results (LR = 0.001) are shown below (LR = 0.003 results are shown in the normalizations comparison section): \

Dropout Training Accuracy on STL10                                                  |  Dropout Training Accuracy on TinyImageNet
:----------------------------------------------------------------------------------:|:----------------------------------------------------------------------------------:
![Dropout Training Accuracy on STL10](./images/DO_STL10_001_acc_test_dropout.png)   |  ![Dropout Training Accuracy on TinyImageNet](./images/DO_TINY_001_acc_test_dropout.png)
Dropout Training Loss on STL10                                                      |  Dropout Training Loss on TinyImageNet
![Dropout Training Loss on STL10](./images/DO_STL10_001_loss_test_dropout.png)      |  ![Dropout Training Loss on TinyImageNet](./images/DO_TINY_001_loss_test_dropout.png)



Batch Normalization Experiments
-

These experiments compare a baseline CNN model to the same model with batch normalization. We implement batch normalization using `torch.nn.BatchNorm2d`. As in the paper, we compare three variants of batch normalization: one with a default LR of 0.001, and two variants with a 5x and 30x increase in learning rate.

### Results

The batch normalized model with default learning rate of 0.001 gets the highest test accuracy across all three datasets.

### Training Loss Per Dataset

|<img width="500px" src="images/CIFAR-100_loss_train.png"/>|
|:--:|
|CIFAR-100|
<br>

|<img width="500px" src="images/STL10_loss_train.png"/>|
|:--:|
|STL-10|
<br>

|<img width="500px" src="images/TINY_loss_train.png"/>|
|:--:|
|TinyImageNet|
<br>


### Test Accuracy After 10 Epochs
| |**CIFAR-100 Test Accuracy**|**STL-10 Test Accuracy**|**TinyImageNet Test Accuracy**|
|:-----:|:-----:|:-----:|:-----:|
|baseline|37.83|59.975|20.13|
|batch\_norm|51.23|62.175|31.89|
|batch\_norm\_5x|36.43|60.1|14.5|
|batch\_norm\_30x|1.76|40.5875|0.82|


Weight Normalization Experiments
-

### Implementation Details

I tested Weight Normalization on PyTorches native implementation as well as my own.  

Weight Normalization is implemented as a wrapper to a layer, it reparameterizes the weight matrix
every forward call using two new parameters, g and v.  v is really v / ||v|| during initialization. 

In order to make Weight Norm work with existing models, the best way to implement it is to 
store parameters v and g in the layer, but use them to recompute the W weight matrix for that layer
anytime the forward() method is called.

### CIFAR100 Test Results

Setup:

- Learning rate of 0.001
- 40 Epochs
- Batch Size of 64

Models:

- Resnet Backbone
- ConvPool CNN C (from the paper)


![My norm vs Torch norm test acc](./images/WNT_testing_accuracy_my_norm_vs_torch_norm_vs_no_norm_0.001%7CAdam%7CCIFAR-100.png)
![My norm vs Torch norm train loss](./images/WNT_training_loss_my_norm_vs_torch_norm_vs_no_norm_0.001%7CAdam%7CCIFAR-100.png)

### Interpretation of results

In all my experiments, weight normalization prevented the model from overfitting and allowed the model
to converge at better test accuracies. 

Interestingly, I expected my implementation to match PyTorches, however, there were significant differences
in the results shown in the graph.  PyTorches implementation has a few slight differences in their forward
call, but they also do not divide the weight matrix by g when initializing the v matrix.  

Although this is probably harmless, I experimented with this myself but the test results were inconclusive, 
so the discrepancy is an unsolved oddity. (one potential explanation that I did not have time to explore is 
that pytorch uses the L2 norm for the weight calculation, I used L1)



# References
- Srivastava, N., et al. 2014, 'Dropout: A Simple Way to Prevent Neural Networks from Overfitting', *Journal of Machine Learning Research*, vol. 15, pp. 1929-1958.
- Ioffe, S., Szegendy, C. 2015, 'Batch Normalization: Accelerating Deep Network Training by Reducing Internal Covariate Shift', *Proceedings of the 32nd International Conference on Machine Learning*, PMLR 37:448-456
- Salimans, T., Kingma, D. 2016, 'Weight Normalization: A Simple Reparameterization to Accelerate Training of Deep Neural Networks', *30th Conference on Neural Information Processing Systems (NIPS 2016)*, pp. 901-909
- Leo Ba, J., et al. 2016, 'Layer Normalization', arXiv:1607.06450, *University of Toronto*.
- Ulyanov, D., et al. 2016, 'Instance Normalization: The Missing Ingredient for Fast Stylization', arXiv:1607.08022
