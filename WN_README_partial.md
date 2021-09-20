Weight Normalization
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
so the discrepancy is an unsolved oddity. 