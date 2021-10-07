# Spatial Transformer Networks in PyTorch

Reference: https://pytorch.org/tutorials/intermediate/spatial_transformer_tutorial.html

Spatial transformer networks are a generalization of differentiable attention to any spatial transformation, and allow a neural network to learn how to perform spatial transformations on the input image in order to enhance the geometric invariance of the model. Project goals:

- Ivestigate if using CoordConv layers instead of standard Conv will help to improve the performance.
- Compare the performance of the new model in evaluation metric and motivate the choice of metrics.
