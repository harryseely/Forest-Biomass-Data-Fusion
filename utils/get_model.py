import torch.nn as nn

from models.ocnn_lenet import OCNN_LeNet
from models.cnn_1d import CNN1D
from models.cnn_2d import CNN2D
from models.modules import Regressor, CatFusionModule


def get_model(cfg):
    assert ((cfg['ocnn_lenet']) or
            (cfg['spec_cnn']) or
            (cfg['terrain_cnn'])), "No model specified in config"

    # Instatiate model dict to store modules
    model = dict()

    # Set the base number of input channels for lidar (XYZ) -> 3
    ocnn_in_channels = 3

    # Set number of input bands
    n_bands = len(cfg['bands'].split(","))

    # Modify number of channels for OCNN if using additional lidar features
    if cfg['ocnn_use_feats']:
        ocnn_in_channels += 5  # Lidar features (intensity, scan angle, return number, classification, number of returns)

    if cfg['use_normals']:
        ocnn_in_channels += 1  # Normals are represented as 1 value

    # Select octree-based architecture
    if cfg['ocnn_lenet']:

        if cfg['ocnn_stages'] == "auto":
            cfg['ocnn_stages'] = cfg['octree_depth'] - 2

        model['lidar_branch'] = OCNN_LeNet(ocnn_stages=cfg['ocnn_stages'], dropout=cfg['ocnn_dropout'],
                                           in_channels=ocnn_in_channels,
                                           out_channels=cfg['n_neurons_final'], nempty=True)

    # Select 1-D CNN architecture
    if cfg['spec_cnn']:

        model['spectral_branch'] = CNN1D(seq_length=n_bands,
                                         in_channels=1,
                                         out_channels=cfg['n_neurons_final'],
                                         dropout=cfg['cnn1d_dropout'],
                                         global_pool=cfg['cnn1d_global_pool'],
                                         n_neurons_1=cfg['cnn1d_n_neurons_1'],
                                         n_neurons_2=cfg['cnn1d_n_neurons_2'],
                                         neuron_mult=cfg['cnn1d_neuron_mult'],
                                         n_layers=cfg['cnn1d_n_layers'],
                                         pooling_type=cfg['cnn1d_pooling_type'],
                                         conv_kernel_size=cfg['cnn1d_conv_kernel_size'],
                                         pool_kernel_size=cfg['cnn1d_pool_kernel_size'],
                                         stride=cfg['cnn1d_stride'],
                                         stride_1=cfg['cnn1d_stride_1'],
                                         dilation_rate=cfg['cnn1d_dilation_rate'],
                                         activation_fn=cfg['cnn1d_activation_fn']
                                         )

    else:
        pass


    # Select terrain CNN architecture
    if cfg['terrain_cnn']:
        model['terrain_branch'] = CNN2D(in_channels=4,
                                    out_channels=cfg['n_neurons_final'],
                                    dropout=cfg['terrain_cnn_dropout'],
                                    global_pool=nn.AdaptiveAvgPool2d,
                                    n_neurons_1=cfg['terrain_cnn_n_neurons_1'],
                                    n_neurons_2=cfg['terrain_cnn_n_neurons_2'],
                                    neuron_mult=cfg['terrain_cnn_neuron_mult'],
                                    n_layers=cfg['terrain_cnn_n_layers'],
                                    pooling_type=cfg['terrain_cnn_pooling_type'],
                                    conv_kernel_size=cfg['terrain_cnn_conv_kernel_size'],
                                    pool_kernel_size=cfg['terrain_cnn_pool_kernel_size'],
                                    stride=cfg['terrain_cnn_stride'],
                                    stride_1=cfg['terrain_cnn_stride_1'],
                                    dilation_rate=cfg['terrain_cnn_dilation_rate'],
                                    activation_fn=cfg['terrain_cnn_activation_fn']
                                    )

    # Build main model --------------------------------------------------------------------------------------------------

    # Build fusion model if using multiple branches/modules
    if len(model.keys()) > 1:
        model = CatFusionModule(model, n_neurons_final=cfg['n_neurons_final'], layer_norm=cfg['fusion_lyr_norm'])


    # if only using 1 branch/module, set model to that branch/module
    else:
        model = list(model.values())[0]

    # Wrap model in regressor module
    model = Regressor(model=model, n_neurons_final=cfg['n_neurons_final'], num_outputs=4, dropout=cfg['dropout_final'])

    print(model)

    return model


if __name__ == "__main__":
    import yaml

    # Read config
    with open(r"config.yaml", "r") as yamlfile:
        cfg = yaml.load(yamlfile, Loader=yaml.FullLoader)

    model = get_model(cfg)

    print(model)
