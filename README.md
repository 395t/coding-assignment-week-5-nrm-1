Normalization Methods - Week 5 Group 2
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

- Resent
- ViT
- ConvPool-CNN-C

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
    

CHART HERE

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
| Instance Normalization | | |

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
| Instance Normalization | | |




