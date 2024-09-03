import torch
import torch.nn as nn
import math

from utils.training_utils import get_activation


class CNN1D(nn.Module):
    def __init__(self,
                 seq_length,
                 in_channels=1,
                 out_channels=1024,
                 n_layers=2,
                 n_neurons_1=16,
                 n_neurons_2=32,
                 neuron_mult=1,
                 global_pool=False,
                 pooling_type='max',
                 dilation_rate=1,
                 conv_kernel_size=3,
                 pool_kernel_size=3,
                 stride=1,
                 stride_1=3,
                 dropout=0.5,
                 activation_fn='relu',
                 groups=1
                 ):
        """
        Model adapted from the repo https://github.com/langnico/GEDI-BDL which is from the following study:
        Global canopy height regression and uncertainty estimation from GEDI LIDAR waveforms with deep ensembles
        by Lang et al. (2021) https://doi.org/10.1016/j.rse.2021.112760

        :param seq_length: length of the input sequence.
        :param in_channels: input channel dimension.
        :param out_channels: number of the output channels .
        :param n_layers: number of convolutional layers.
        :param n_neurons_1: number of neurons in the first convolutional layer.
        :param n_neurons_2: number of neurons in the second convolutional layer.
        :param neuron_mult: multiplier for the number of neurons in the subsequent convolutional layers.
        :param global_pool: Boolean; whether to use Adaptive Average global pooling
        :param pooling_type: type of pooling to do after each layer. Options are 'max' or 'avg'
        :param dilation_rate: must be >= 1. Defaults to 1 (no dilation).
        :param conv_kernel_size: must be >= 1. Defaults to 3.
        :param stride: must be >= 1. Defaults to 2.
        :param dropout: dropout rate.
        :param activation_fn: activation function to use.
        :param groups: see the PyTorch documentation for nn.Conv1d 'groups' parameter

        """
        super(CNN1D, self).__init__()

        # Set activation function
        self.activation = get_activation(activation_fn)

        if global_pool:
            self.global_pool = nn.AdaptiveAvgPool1d
        else:
            self.global_pool = None

        out_features = [n_neurons_1, n_neurons_2]
        if n_layers > 2:
            for _ in range(n_layers - 2):
                out_features.append(n_neurons_2 * neuron_mult)

        out_features = out_features[:n_layers]

        if global_pool:
            final_fc_features = out_features[-1]
        else:
            final_fc_features = out_features[-1] * math.ceil(seq_length / 2)

        # Loop through each layer and apply convolution, batch norm, activation, and pooling
        layers = list()
        for i in range(n_layers):

            # Set number of input channels
            in_channels = in_channels if i == 0 else out_features[i - 1]

            # Set the stride
            lry_stride = stride_1 if i == 0 else stride

            layers.append(nn.Conv1d(in_channels=in_channels, out_channels=out_features[i], kernel_size=conv_kernel_size,
                                    padding=1, dilation=dilation_rate, groups=groups))

            layers.append(nn.BatchNorm1d(num_features=out_features[i]))

            layers.append(self.activation)

            # Apply pooling
            if pooling_type == 'max':
                layers.append(nn.MaxPool1d(kernel_size=pool_kernel_size, stride=lry_stride, padding=1))
            elif pooling_type == 'avg':
                layers.append(nn.AvgPool1d(kernel_size=pool_kernel_size, stride=lry_stride, padding=1))
            else:
                raise ValueError(f'Invalid pooling type: {pooling_type}')

        self.conv_layers = nn.Sequential(*layers)

        if self.global_pool is not None:
            self.global_pool = self.global_pool(output_size=1)

        self.dropout = nn.Dropout(p=dropout)

        self.fc = nn.Linear(in_features=final_fc_features, out_features=out_channels)

    def forward(self, batch):
        """
        Data enters network with shape (batch_size, channels, sequence_length)
        :param batch:
        :return:
        """
        x = batch['spectral']

        x = self.conv_layers(x)

        if self.global_pool:
            x = self.global_pool(x)

        x = self.dropout(x)

        x = x.flatten(start_dim=1)

        x = self.fc(x)

        return x
