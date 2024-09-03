# Pytorch Implementation: https://github.com/octree-nn/ocnn-pytorch
# Citation: Wang, Peng-Shuai, et al. "O-cnn: Octree-based convolutional neural networks for 3d shape analysis." ACM Transactions On Graphics (TOG) 36.4 (2017): 1-11.

import torch
import ocnn


class OCNN_LeNet(torch.nn.Module):
    r''' Octree-based LeNet for classification.
  '''

    def __init__(self, ocnn_stages, dropout, in_channels, out_channels: int, nempty: bool = False):
        super().__init__()

        # Set up model
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.stages = ocnn_stages
        self.nempty = nempty
        self.dropout = dropout
        channels = [in_channels] + [2 ** max(i + 7 - self.stages, 2) for i in range(self.stages)]

        self.convs = torch.nn.ModuleList([ocnn.modules.OctreeConvBnRelu(
            channels[i], channels[i + 1], nempty=nempty) for i in range(self.stages)])
        self.pools = torch.nn.ModuleList([ocnn.nn.OctreeMaxPool(
            nempty) for i in range(self.stages)])
        self.octree2voxel = ocnn.nn.Octree2Voxel(self.nempty)
        self.header = torch.nn.Sequential(
            torch.nn.Dropout(p=self.dropout),  # drop1
            ocnn.modules.FcBnRelu(64 * 64, 128),  # fc1
            torch.nn.Dropout(p=self.dropout),  # drop2
            torch.nn.Linear(128, out_channels))  # fc2

    def forward(self, batch):
        r''''''

        # Modified from original model
        octree = batch['octree']

        # Get model input features from octree object
        data = octree.get_input_feature()
        # Get the depth of the octree
        depth = octree.depth

        for i in range(self.stages):
            d = depth - i
            data = self.convs[i](data, octree, d)
            data = self.pools[i](data, octree, d)
        data = self.octree2voxel(data, octree, depth - self.stages)
        data = self.header(data)

        return data
