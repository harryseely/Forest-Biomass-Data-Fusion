import torch
import torch.nn as nn

from utils.training_utils import get_activation

class CNN2D(nn.Module):
    def __init__(self,
                 in_channels=1,
                 out_channels=1024,
                 n_layers=2,
                 n_neurons_1=16,
                 n_neurons_2=32,
                 neuron_mult=1,
                 global_pool=nn.AdaptiveAvgPool2d,
                 pooling_type='max',
                 dilation_rate=1,
                 conv_kernel_size=3,
                 pool_kernel_size=3,
                 stride=1,
                 stride_1=3,
                 dropout=0.5,
                 activation_fn='relu'
                 ):
        """
        Model based on the publication by Kirkwood et al. (2022):
        "Bayesian Deep Learning for Spatial Interpolation in the Presence of Auxiliary Information"
        https://doi.org/10.1007/s11004-021-09988-0

        :param in_channels: input channel dimension.
        :param out_channels: number of the output channels .
        :param n_layers: number of convolutional layers.
        :param n_neurons_1: number of neurons in the first convolutional layer.
        :param n_neurons_2: number of neurons in the second convolutional layer.
        :param neuron_mult: multiplier for the number of neurons in the subsequent convolutional layers.=
        :param global_pool: the global pooling to use before the fully connected layer.
        :param pooling_type: type of pooling to do after each layer. Options are 'max' or 'avg'
        :param dilation_rate: must be >= 1. Defaults to 1 (no dilation).
        :param conv_kernel_size: must be >= 1. Defaults to 3.
        :param pool_kernel_size: must be >= 1. Defaults to 3.
        :param stride: must be >= 1. Defaults to 2.
        :param stride_1: stride used specicically in first layer must be >= 1. Defaults to 3.
        :param dropout: dropout rate.
        :param activation_fn: activation function to use.

        """
        super(CNN2D, self).__init__()

        # Set activation function
        self.activation = get_activation(activation_fn)

        out_features = [n_neurons_1, n_neurons_2]
        if n_layers > 2:
            for _ in range(n_layers - 2):
                out_features.append(n_neurons_2 * neuron_mult)

        out_features = out_features[:n_layers]

        # Loop through each layer and apply convolution, batch norm, activation, and pooling
        layers = list()
        for i in range(n_layers):

            #Set number of input channels
            in_channels = in_channels if i == 0 else out_features[i - 1]

            #Set the stride
            lry_stride = stride_1 if i == 0 else stride

            layers.append(
                nn.Conv2d(in_channels=in_channels, out_channels=out_features[i], kernel_size=conv_kernel_size,
                          padding=1, dilation=dilation_rate))
            layers.append(nn.BatchNorm2d(num_features=out_features[i]))
            layers.append(self.activation)

            # Apply pooling
            if pooling_type == 'max':
                layers.append(nn.MaxPool2d(kernel_size=pool_kernel_size, stride=lry_stride, padding=1))
            elif pooling_type == 'avg':
                layers.append(nn.AvgPool2d(kernel_size=pool_kernel_size, stride=lry_stride, padding=1))
            else:
                raise ValueError(f'Invalid pooling type: {pooling_type}')

        self.conv_layers = nn.Sequential(*layers)
        self.global_pool = global_pool(output_size=1)
        self.dropout = nn.Dropout(p=dropout)
        self.fc = nn.Linear(in_features=out_features[-1], out_features=out_channels)

    def forward(self, batch):
        x = batch['terrain']
        x = self.conv_layers(x)
        x = self.global_pool(x)
        x = self.dropout(x)
        x = x.flatten(start_dim=1)
        x = self.fc(x)

        return x