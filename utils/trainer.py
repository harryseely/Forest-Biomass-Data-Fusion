import pytorch_lightning as pl
import torch
from torchmetrics import R2Score
from torchmetrics.functional import r2_score, mean_squared_error
import os

# My code
from utils.get_model import get_model
from utils.training_utils import LossFunction, forward_pass
from utils.data_utils import re_convert_to_Mg_ha, update_z_score_conversion_info


class LitModel(pl.LightningModule):
    def __init__(self, config, return_id_or_bbox_center='PlotID'):
        """
        Lightning module for training and inference.

        predict_step() generates output tensors that have the following indices for each biomass component:
            0 -> bark
            1 -> branch
            2 -> foliage
            3 -> wood

        :param config: dictionary containing many config parameters relating to hyperparameters, filepaths, etc.
        :param return_id_or_bbox_center: trainer.predict() returns, in addition to a list of predictions,
        a list of plot ids or bounding boxes. Can be either 'plotid' or bbox_center'. Default is 'plotid'.
        """
        super(LitModel, self).__init__()

        # Whether to return PlotID in predict step
        self.return_id_or_bbox_center = return_id_or_bbox_center

        # Attach all arguments to lightning module
        self.config = config

        # Update mean and sd and save for z-score conversion
        if self.config['z_score']:
            self.z_info = update_z_score_conversion_info(os.path.join(config['data_dir'], config['dataset']))

        # Set up model
        self.model = get_model(config)

        # Instantiate loss function
        self.loss_fn = LossFunction(loss_fn_name=config['loss_function_type'])

        # Set up evaluation metrics with torchmetrics
        self.train_r2 = R2Score(num_outputs=4, adjusted=0, multioutput='uniform_average', dist_sync_on_step=False)
        self.val_r2 = R2Score(num_outputs=4, adjusted=0, multioutput='uniform_average', dist_sync_on_step=False)
        self.test_r2 = R2Score(num_outputs=4, adjusted=0, multioutput='uniform_average', dist_sync_on_step=False)

    def training_step(self, batch, batch_idx):
        # Forward pass
        pred = forward_pass(self.model, batch)

        # Calculate loss
        train_loss = self.loss_fn(pred=pred, y=batch['target'])

        if self.config['z_score']:
            # Convert pred and y from z-score to Mg/ha value
            pred = re_convert_to_Mg_ha(self.z_info, z_components_arr=pred)
            y = re_convert_to_Mg_ha(self.z_info, z_components_arr=batch['target'])
        else:
            y = batch['target']

        train_r2 = self.train_r2(pred, y)

        # Log metrics
        self.log("train_loss", value=train_loss, batch_size=self.config['batch_size'],
                 on_step=False, on_epoch=True, prog_bar=True, sync_dist=True)
        self.log("train_r2", value=train_r2, batch_size=self.config['batch_size'],
                 on_step=False, on_epoch=True, prog_bar=True, sync_dist=True)

        return train_loss

    def validation_step(self, batch, batch_idx):
        # Forward pass
        pred = forward_pass(self.model, batch)

        # Calculate loss
        val_loss = self.loss_fn(pred=pred, y=batch['target'])

        if self.config['z_score']:
            # Convert pred and y from z-score to Mg/ha value
            pred = re_convert_to_Mg_ha(self.z_info, z_components_arr=pred)
            y = re_convert_to_Mg_ha(self.z_info, z_components_arr=batch['target'])
        else:
            y = batch['target']

        val_r2 = self.val_r2(pred, y)

        # Log metrics
        self.log("val_loss", value=val_loss, batch_size=self.config['batch_size'],
                 on_step=False, on_epoch=True, prog_bar=True, sync_dist=True)
        self.log("val_r2", value=val_r2, batch_size=self.config['batch_size'],
                 on_step=False, on_epoch=True, prog_bar=True, sync_dist=True)

        return val_loss

    def test_step(self, batch, batch_idx):
        # Forward pass
        pred = forward_pass(self.model, batch)

        # Calculate loss
        test_loss = self.loss_fn(pred=pred, y=batch['target'])

        if self.config['z_score']:
            # Convert pred and y from z-score to Mg/ha value
            pred = re_convert_to_Mg_ha(self.z_info, z_components_arr=pred)
            y = re_convert_to_Mg_ha(self.z_info, z_components_arr=batch['target'])
        else:
            y = batch['target']

        # Get loss of total AGB
        tree_pred = pred[:, 0] + pred[:, 1] + pred[:, 2] + pred[:, 3]
        tree_obs = y[:, 0] + y[:, 1] + y[:, 2] + y[:, 3]

        # Calculate metrics for component
        comp_list = ['bark', 'branch', 'foliage', 'wood']
        idx_list = [0, 1, 2, 3]
        test_metric_dict = dict()
        for comp, idx in zip(comp_list, idx_list):
            test_metric_dict[comp + "_r2"] = r2_score(preds=pred[:, idx], target=y[:, idx])
            test_metric_dict[comp + "_rmse"] = torch.sqrt(mean_squared_error(preds=pred[:, idx], target=y[:, idx]))

        # Calculate metrics for tree and overall
        test_metric_dict['tree_r2'] = r2_score(preds=tree_pred, target=tree_obs)
        test_metric_dict['overall_r2'] = r2_score(preds=pred, target=y, adjusted=0, multioutput='uniform_average')
        test_metric_dict['tree_mse'] = torch.sqrt(mean_squared_error(preds=tree_pred, target=tree_obs))
        test_metric_dict['overall_mse'] = torch.sqrt(mean_squared_error(preds=pred, target=y))

        # Log metrics
        self.log_dict(test_metric_dict, batch_size=self.config['batch_size'])

        return test_loss

    def predict_step(self, batch, batch_idx, dataloader_idx=0):

        """
        Prediction step currently set up to use monte carlo dropout.

        **Note, for now, this is only meant to be implemented using 1 GPU
        :param batch: batch data
        :param batch_idx: batch index
        :param dataloader_idx: dataloader index. Only required if using mulitple data loaders for inference.
        :return: predicted value converted to original units (Mg/ha
        """

        # Forward pass
        pred = forward_pass(self.model, batch)

        # Get the index of the point cloud that contains no points
        if self.config['ocnn_lenet']:
            indices = [idx for idx, contains_points in enumerate(batch['contains_points']) if contains_points == False]
        else:
            indices = []

        if self.config['z_score']:
            # Convert pred from z-score to Mg/ha value
            pred = re_convert_to_Mg_ha(self.z_info, z_components_arr=pred)

        # For indices where point cloud has no points, assign -9999 to the corresponding pred output tensors
        if len(indices) > 0:
            for idx in indices:
                pred[idx] = torch.tensor([-9999, -9999, -9999, -9999])

        #Set output list
        out_list = [pred]

        # Return batch predictions and, plot id or bbox, or just the pred
        if self.return_id_or_bbox_center.lower() == 'plotid':
            out_list.append(batch['PlotID'])

        elif self.return_id_or_bbox_center.lower() == 'bbox_center':
            out_list.append(batch['bbox_center'])

        else:
            raise ValueError(
                f"return_id_or_bbox_center must be either 'PlotID', 'bbox', or None not {self.return_id_or_bbox_center}")

        return out_list

    def configure_optimizers(self):

        if self.config['cawr_t_0'] == 'num_epochs':
            self.config['cawr_t_0'] = self.config['num_epochs']

        optimizer = torch.optim.AdamW(params=self.model.parameters(), lr=self.config['lr'],
                                      weight_decay=self.config['weight_decay'])
        lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer,
                                                                            T_0=self.config['cawr_t_0'],
                                                                            T_mult=self.config['cawr_t_mult']
                                                                            )
        return [optimizer], [lr_scheduler]
