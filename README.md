# Spatial Transformer Networks in PyTorch

## References

DeepMind paper: https://arxiv.org/abs/1506.02025.

Uber AI paper: https://arxiv.org/pdf/1807.03247.pdf.

PyTorch tutorial: https://pytorch.org/tutorials/intermediate/spatial_transformer_tutorial.html.

Pytorch implementation of CoordConv: https://github.com/walsvid/CoordConv.


## Description

Spatial transformer networks are a generalization of differentiable attention to any spatial transformation. Spatial transformer networks (STN for short) allow a neural network to learn how to perform spatial transformations on the input image in order to enhance the geometric invariance of the model. For example, it can crop a region of interest, scale and correct the orientation of an image. It can be a useful mechanism because CNNs are not invariant to rotation and scale and more general affine transformations. 

Goals of the project:

1. Investigate if using CoordConv layers instead of standard Conv will help to improve the performance.
2. Compare the performance of the new model in evaluation metric and motivate the choice of metrics.
3. Explore new ideas that might achieve better performance than conventional STNs.

## Installation

- [Docker](https://docs.docker.com/get-docker)

## Usage

Set up the environment:
```sh
docker build -t pytorch-stn . 
```

Train and test the STN with different layers:
```sh
docker run -v "$(pwd):/app" pytorch-stn --layer='conv' --epochs=50
```
```sh
docker run -v "$(pwd):/app" pytorch-stn --layer='coordconv' --epochs=50
```

Output images: 
- `imgs/stn.png`: visualize the batch of input images and the corresponding transformed batch using STN
- `imgs/cm.png`: confusion matrix where number of predictions are summarized with count values.

## Experiments

The proposed CoordConv layer is a simple extension to the standard convolutional layer. Convolutional layers are used in a lot of applications because they often work well, perhaps due to some combination of three factors: 
- they have relatively few learned parameters.
- they are fast to compute on modern GPUs.
- they learn a function that is translation invariant. 

Following figure shows a comparison of 2D Conv and CoordConv layers.

![alt text](https://github.com/vicsesi/Pytorch-STN/blob/main/imgs/layers.png?raw=true)

Uber AI paper suggest that including CoordConv layers can boost the performance. In order to verify this hypothesis, we will compare the performance using Conv and CoordConv layers and training the models during 50 epochs. We will evaluate the accuracy for each number in MNIST dataset, and the average loss and the accuracy for the whole test set. Following tables shows the results:

| Layer | 0 | 1 | 2 | 3 | 4 | 5 | 6 | 7 | 8 | 9 |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| Conv | 99% | 99% | 99% | 99% | 99% | 98% | 99% | 98% | 99% | 98% |
| CoordConv | 99% | 99% | 99% | 99% | 98% | 98% | 98% | 99% | 99% | 98% |

| Layer | Average loss | Accuracy |
| :---: | :---: | :---: |
| Conv | 0.0296 | 9921/10000 (99%) |
| CoordConv | 0.0312 | 9908/10000 (99%) |

As we can see on the previous tables, the performances using Conv and CoordConv layers are pretty similar. We will compute the confusion matrix in order to summarize the predictions broken down by each number.

| Confusion Matrix Conv Layer |  Confusion Matrix CoordConv Layer |  
| :-------------------------:|:-------------------------:
| ![alt text](https://github.com/vicsesi/Pytorch-STN/blob/main/imgs/cm_conv_50.png?raw=true) |  ![alt text](https://github.com/vicsesi/Pytorch-STN/blob/main/imgs/cm_coordconv_50.png?raw=true) |

For this image classification problem, using the CoordConv layer doesn't improve the performance in classification task. Although the previous tables shows that the accuracy is slightly worse in predictions with CoordConv layer during 50 training epochs, we've also evaluated the performance with less number of training epochs. 

All of the experiments shown that the accuracy does not improve considerably using CoordConv layers. In image classification we don't expect see much improvement, because Conv layers are actually designed to be spatially invariant.

## Using Leaky ReLU activation function

We will try to boost the performance, using Leaky ReLU activation function instead of ReLU in the spatial transformer network. We will include CoordConv layers as well. The derivative of Leaky ReLU is not a 0 in the negative part, and this activation function have a little slope to allow the gradients to flow on. Let's verify if this condition could be a benefit to improve the performance. We will evaluate the performance following the same methodology than the previous experiments.

Command to reproduce the experiment:

```sh
docker run -v "$(pwd):/app" pytorch-stn --function='leakyrelu' --epochs=50
```

| Activation | 0 | 1 | 2 | 3 | 4 | 5 | 6 | 7 | 8 | 9 |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| Leaky ReLU | 99% | 99% | 99% | 99% | 99% | 98% | 99% | 99% | 99% | 98% |

| Activation | Average loss | Accuracy |
| :---: | :---: | :---: |
| Leaky ReLU  | 0.0293 | 9913/10000 (99%) |

| Confusion Matrix Conv Layer |  Confusion Matrix Leaky ReLU Activation |  
| :-------------------------:|:-------------------------:
| ![alt text](https://github.com/vicsesi/Pytorch-STN/blob/main/imgs/cm_conv_50.png?raw=true) |  ![alt text](https://github.com/vicsesi/Pytorch-STN/blob/main/imgs/cm_leakyrelu_50.png?raw=true) |

We observed that using Leaky ReLU activation functions in STN, doesn't improve the performance in classification task. Change the ReLU activation function by Leaky ReLU, doesn't affect significantly to the performance.
