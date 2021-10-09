import torch
import torch.nn as nn
from addcoords2d import AddCoords2d
import torch.nn.modules.conv as conv


class CoordConv2d(conv.Conv2d):

    def __init__(self, in_channels, out_channels, kernel_size):

        super(CoordConv2d, self).__init__(in_channels, out_channels, kernel_size)
        self.addcoords2d = AddCoords2d()
        self.conv = nn.Conv2d(in_channels + 2, out_channels, kernel_size)

    def forward(self, input_tensor):

        out = self.addcoords2d(input_tensor)
        out = self.conv(out)
        return out
