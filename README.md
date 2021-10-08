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

Train and test the STN:
```sh
docker run -v "$(pwd):/app" pytorch-stn --layer='conv' --epochs=20
```

Inputs arguments:

- `--layer`: string, options: {`conv`, `coordconv`}.
- `--epochs`: integer, must be a positive number.

Output images: 
- `imgs/stn.png`: visualize the batch of input images and the corresponding transformed batch using STN
- `imgs/cm.png`: confusion matrix where number of predictions are summarized with count values.

## Experiments

The proposed CoordConv layer is a simple extension to the standard convolutional layer. Convolutional layers are used in a myriad of applications because they often work well, perhaps due to some combination of three factors: 
- they have relatively few learned parameters.
- they are fast to compute on modern GPUs.
- they learn a function that is translation invariant. 

Following figure shows a comparison of 2D Conv and CoordConv layers.

![alt text](https://github.com/vicsesi/Pytorch-STN/blob/main/imgs/layers.png?raw=true)

Uber AI paper suggest that including CoordConv layers can boost the performance. In order to verify this hypothesis, we will compare the performance using Conv and CoordConv layers, with 20 epochs during the training. We will evaluate the accuracy for each number in MNIST dataset, and the average loss and accuracy for the whole set. Following tables shows the results:

| Layer | 0 | 1 | 2 | 3 | 4 | 5 | 6 | 7 | 8 | 9 |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| Conv | 99% | 99% | 98% | 99% | 99% | 98% | 98% | 97% | 98% | 98% |
| CoordConv | % | % | % | % | % | % | % | % | % | % |

| Layer | Average loss | Accuracy |
| :---: | :---: | :---: |
| Conv | 0.0428 | 9880/10000 (99%) |
| CoordConv | | /10000 (%) |

| Confusion Matrix Conv |  Confusion Matrix CoordConv |  
| :-------------------------:|:-------------------------:
| ![alt text](https://github.com/vicsesi/Pytorch-STN/blob/main/imgs/cm_conv_20.png?raw=true) |  ![alt text](https://github.com/vicsesi/Pytorch-STN/blob/main/imgs/cm_coordconv_20.png?raw=true) |
