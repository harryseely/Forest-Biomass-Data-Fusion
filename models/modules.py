import torch
import torch.nn as nn


class CatFusionModule(torch.nn.Module):
    def __init__(self, model_dict, n_neurons_final, layer_norm):
        """
        Fusion module for combining the outputs of the individual models.

        Liu et al. (2023) https://doi.org/10.1109/TGRS.2021.3114460
        Kirkwood et al. (2022) https://doi.org/10.1007/s11004-021-09988-0

        """
        super().__init__()

        self.mod_nms = list(model_dict.keys())

        # Following Liu et al. (2023), apply layer normalization to the output tensors of each model

        # Define a dictionary to store the modules
        modules = {}

        for mod_nm in self.mod_nms:
            if layer_norm:
                lyrs = list()
                lyrs.append(model_dict[mod_nm])
                lyrs.append(nn.LayerNorm(n_neurons_final))
                modules[mod_nm] = nn.Sequential(*lyrs)
            else:
                modules[mod_nm] = model_dict[mod_nm]

        # Assign the modules to instance attributes
        if 'lidar_branch' in self.mod_nms:
            self.lidar_branch = modules.get("lidar_branch")

        if 'spectral_branch' in self.mod_nms:
            self.spectral_branch = modules.get("spectral_branch")

        if 'terrain_branch' in self.mod_nms:
            self.terrain_branch = modules.get("terrain_branch")

        self.out_fc = nn.Linear(n_neurons_final * len(self.mod_nms), n_neurons_final)

    def forward(self, batch):

        outs = list()

        if 'lidar_branch' in self.mod_nms:
            outs.append(self.lidar_branch(batch))

        if 'spectral_branch' in self.mod_nms:
            outs.append(self.spectral_branch(batch))

        if 'terrain_branch' in self.mod_nms:
            outs.append(self.terrain_branch(batch))

        # Concatenate outputs
        cat_out = torch.cat(outs, dim=1)

        out = self.out_fc(cat_out)

        return out


class Regressor(torch.nn.Module):
    """
    Regressor wrapper for individual or fusion model.

    """

    def __init__(self, model, n_neurons_final, num_outputs, dropout=0.5, activation_fn="relu"):
        """

        :param model: Input pytorch model that returns a tensor of shape batch size x n_neurons_final
        :param n_neurons_final: Number of neurons in the final layer of the model
        :param num_outputs: Number of outputs from the regressor layer
        :param dropout: Dropout probability
        :param activation_fn: activation function for the regressor layer
        """
        super().__init__()
        self.model = model
        self.dropout = dropout
        self.num_outputs = num_outputs

        if activation_fn == "relu":
            activation_fn = nn.ReLU()
        elif activation_fn == "leaky_relu":
            activation_fn = nn.LeakyReLU()

        # Create a list of main_head for each output
        self.main_heads = nn.ModuleList([nn.Sequential(
            nn.Linear(n_neurons_final, n_neurons_final),
            nn.BatchNorm1d(n_neurons_final),
            activation_fn,
            nn.Dropout(p=self.dropout),
            nn.Linear(n_neurons_final, 64),
            nn.BatchNorm1d(64),
            activation_fn,
            nn.Dropout(p=self.dropout),
            nn.Linear(64, 1),
        ) for _ in range(self.num_outputs)])

    def forward(self, batch):
        x = self.model(batch)

        # Apply each main_head in the list to the input x
        pred = torch.stack([main_head(x) for main_head in self.main_heads], dim=1)
        pred = pred.squeeze()

        return pred
