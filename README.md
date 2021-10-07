# Spatial Transformer Networks in PyTorch

Spatial transformer networks are a generalization of differentiable attention to any spatial transformation. Spatial transformer networks (STN for short) allow a neural network to learn how to perform spatial transformations on the input image in order to enhance the geometric invariance of the model. Project goals:

- Ivestigate if using CoordConv layers instead of standard Conv will help to improve the performance of the baseline.
- Compare the performance of the new model in evaluation metric and motivate the choice of metrics.
