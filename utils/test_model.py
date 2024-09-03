import wandb
import os
import yaml
import pytorch_lightning as pl
from itertools import chain
import geopandas as gpd
from shapely.geometry import Point
import pandas as pd
from pathlib import Path

# My code
from utils.trainer import LitModel
from utils.dataset import BiomassDataModule
from utils.generate_obs_pred_plots import get_metrics_make_plots

def model_predict(lightning_model, data_module, ckpt_file, device_id, return_id_or_bbox_center, crs=None,
                  progress_bar=False):
    """
    Loads a pytorch lightning model from a checkpoint directory and generates predictions for a given lidar tile or
    sample plot point cloud depending on the value of the return_id_or_bbox_center.

    :param lightning_model: pytorch lightning model object
    :param data_module: pytorch lightning data module object
    :param device_id: integer indicating GPU device to use (e.g., 0)
    :param return_id_or_bbox_center: depends on use case, whether to return plot ids or bounding box centers.
    Can be either 'plotid' or 'bbox_center'
    :param crs: coordinate reference system of input lidar tile and generated raster
    config file loaded from the checkpoint directory
    :param progress_bar: whether to display a progress bar during pytorch lightning prediction
    :return: list of predictions for each bounding box
    """

    # Instatite pl trainer object
    trainer = pl.Trainer(accelerator="gpu",
                         devices=[device_id],
                         logger=None,
                         enable_checkpointing=False,
                         enable_progress_bar=progress_bar)

    # Generate predictions
    out_list = trainer.predict(lightning_model, datamodule=data_module, ckpt_path=ckpt_file,
                               return_predictions=True)

    # Seperate the predicted tensors from the bounding boxe centres
    preds = [i[0] for i in out_list]

    #For indexing output list
    plotid_bbox_idx = 1

    # Whether to use the plot id or bbox center
    if return_id_or_bbox_center == 'bbox_center':

        bbox_centers = [i[plotid_bbox_idx] for i in out_list]

        bbox_centers = list(chain(*bbox_centers))

        # Convert bbox centers to geopandas points
        points = [Point(coordinate) for coordinate in bbox_centers]

        # Convert bbox centers to geopandas dataframe
        df = gpd.GeoDataFrame(geometry=points, crs=crs)

    elif return_id_or_bbox_center == 'plotid':

        plot_ids = [i[plotid_bbox_idx] for i in out_list]
        plot_ids = list(chain(*plot_ids))

        # Convert plot ids to geopandas dataframe
        df = pd.DataFrame(plot_ids, columns=['PlotID'])

    else:
        raise ValueError(
            f"return_id_or_bbox_center must be either 'bbox_center' or 'plotid' {return_id_or_bbox_center} was provided")

    # Flatten lists of predictions
    preds = list(chain(*preds))
    preds = [i.detach().cpu().numpy() for i in preds]

    # Add the biomass predictions to the geopandas dataframe
    df['bark_pred'] = [i[0] for i in preds]
    df['branch_pred'] = [i[1] for i in preds]
    df['foliage_pred'] = [i[2] for i in preds]
    df['wood_pred'] = [i[3] for i in preds]

    return df


def test_model(save_dir, logger=None, fig_out_dir=r"", save_plots_to_wandb=True,
               view_plots=True):
    """
    Test a model and log the results to wandb
    :param save_dir: directory where config and model checkpoint are stored
    :param logger: wandb logger object
    :param fig_out_dir: directory to save plots
    :param save_plots_to_wandb: boolean, whether to save plots to wandb
    :param view_plots: boolean, whether to view plots immediately
    :return:
    """
    # If you do not have a trainer and model available, provide the directory where config and model checkpoint are stored
    cfg_files = list(Path(save_dir).glob("*" + 'yaml'))
    ckpt_files = list(Path(save_dir).glob("*" + 'ckpt'))

    assert len(cfg_files) == 1, print(f"Multiple config files present:\n{cfg_files}")
    assert len(ckpt_files) == 1, print(f"Multiple checkpoint files present:\n{ckpt_files}")

    cfg_file = cfg_files[0]
    ckpt_file = ckpt_files[0]

    with open(cfg_file, "r") as yamlfile:
        cfg = yaml.load(yamlfile, Loader=yaml.FullLoader)

    # Rebuild model for testing uprojesing correct config
    model = LitModel(cfg)

    # Set up data module with same params
    data_module = BiomassDataModule(cfg=cfg)

    df = model_predict(
        lightning_model=model,
        data_module=data_module,
        ckpt_file=ckpt_file,
        device_id=0,
        return_id_or_bbox_center="plotid",
        crs=None,
        progress_bar=True
    )

    # Load reference df and join
    ref_data_path = os.path.join(cfg['data_dir'], cfg['dataset'], 'biomass_labels.csv')
    ref_df = pd.read_csv(ref_data_path, sep=",", header=0)

    #Ensure PlotID is a string
    ref_df["PlotID"] = ref_df["PlotID"].astype(str)
    df = df.merge(ref_df, how="left", on="PlotID")

    # rename columns to obs
    df = df.rename(columns={

        # Observed
        "bark_Mg_ha": "bark_Mg_ha_obs",
        "branch_Mg_ha": "branch_Mg_ha_obs",
        "foliage_Mg_ha": "foliage_Mg_ha_obs",
        "wood_Mg_ha": "wood_Mg_ha_obs",
        "total_Mg_ha": "tree_Mg_ha_obs",

        # Predicted
        "wood_pred": "wood_Mg_ha_pred",
        "bark_pred": "bark_Mg_ha_pred",
        "branch_pred": "branch_Mg_ha_pred",
        "foliage_pred": "foliage_Mg_ha_pred"
    })

    # Calculate total pred
    df["tree_Mg_ha_pred"] = df["wood_Mg_ha_pred"] + df["bark_Mg_ha_pred"] + df["branch_Mg_ha_pred"] + df[
        "foliage_Mg_ha_pred"]

    # Compute residuals
    df["wood_Mg_ha_resid"] = df["wood_Mg_ha_obs"] - df["wood_Mg_ha_pred"]
    df["bark_Mg_ha_resid"] = df["bark_Mg_ha_obs"] - df["bark_Mg_ha_pred"]
    df["branch_Mg_ha_resid"] = df["branch_Mg_ha_obs"] - df["branch_Mg_ha_pred"]
    df["foliage_Mg_ha_resid"] = df["foliage_Mg_ha_obs"] - df["foliage_Mg_ha_pred"]
    df["tree_Mg_ha_resid"] = df["tree_Mg_ha_obs"] - df["tree_Mg_ha_pred"]

    metrics_df = get_metrics_make_plots(df=df,
                                        datetime="",
                                        target_label="Mg_ha",
                                        axis_lab="Biomass Mg/ha",
                                        fig_out_dir=fig_out_dir,
                                        view_plots=view_plots
                                        )

    #Export the predicted values to a csv
    df_out_fpath = os.path.join(save_dir, os.path.basename(os.path.dirname(ckpt_file)) + "-test_set_preds.csv")
    df.to_csv(df_out_fpath, index=False)

    # Log metrics to wandb
    if logger is not None:

        #Record the number of parameters in the model
        n_parameters = sum(p.numel() for p in model.parameters())
        wandb.config.update({"n_parameters": n_parameters}, allow_val_change=True)

        # Convert the df into a dict for logging
        metrics_dict = metrics_df.to_dict()

        metrics_dict_full = {}

        # Rename keys
        for metric in metrics_dict.keys():
            for comp in metrics_dict[metric].keys():
                metrics_dict_full[f"{comp}_{metric}".replace("_Mg_ha", "")] = metrics_dict[metric][comp]

        #Save performance metrics to wandb
        wandb.config.update(metrics_dict_full, allow_val_change=True)

        if save_plots_to_wandb:
            # Generate plots in wandb as well
            table = wandb.Table(data=df)

            for comp in ["bark", "branch", "foliage", "wood", "tree"]:
                wandb.log(
                    {f"{comp}_resid_plot": wandb.plot.scatter(table,
                                                              x=f"{comp}_Mg_ha_obs",
                                                              y=f"{comp}_Mg_ha_resid",
                                                              title=f"{comp} Residuals"
                                                              )})

                wandb.log(
                    {f"{comp}_obs_pred_plot": wandb.plot.scatter(table,
                                                                 x=f"{comp}_Mg_ha_obs",
                                                                 y=f"{comp}_Mg_ha_pred",
                                                                 title=f"{comp} Obs-Pred"
                                                                 )})