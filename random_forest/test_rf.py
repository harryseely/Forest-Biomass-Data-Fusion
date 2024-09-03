# Modules
import torch
from joblib import load
import numpy as np
import pandas as pd
import sklearn.metrics as metrics
from math import sqrt
from matplotlib import pyplot as plt

# My scripts
from utils.data_utils import re_convert_to_Mg_ha
from utils.data_utils import update_z_score_conversion_info

# Supress warnings
import warnings

warnings.filterwarnings("ignore")


def convert_array_to_df(arr, plot_id, cfg):
    """
    Function converts an input numpy array to pandas df. Ensure shape of input array matches critria described below.
    Also calculates total AGB from components and includes calculation of residuals.
    :param arr: input array of shape (obs, pred). If predicting biomass comps, col order must be bark, branch, foliage, wood
    :param plot_id: list of plot ids that match with each obs/pred row in input arr
    :return:
    """
    if cfg['rf_target'] == "total_agb":  # Fill component
        df = pd.DataFrame(arr, columns=[f'tree_obs', f'tree_pred'], index=plot_id)
        df[f"tree_resid"] = df[f"tree_obs"] - df[f"tree_pred"]

    elif cfg['rf_target'] == "biomass_comps":
        # Convert to data frame
        df = pd.DataFrame(arr,
                          columns=[f'bark_obs', f'branch_obs',
                                   f'foliage_obs',
                                   f'wood_obs',
                                   f'bark_pred', f'branch_pred',
                                   f'foliage_pred',
                                   f'wood_pred'],
                          index=plot_id)

        # Add observed/predicted total biomass columns to df
        df[f"tree_obs"] = df[f"bark_obs"] + df[f"branch_obs"] + df[f"foliage_obs"] + df[f"wood_obs"]
        df[f"tree_pred"] = df[f"bark_pred"] + df[f"branch_pred"] + df[
            f"foliage_pred"] + df[f"wood_pred"]

        # Get residuals
        df[f"tree_resid"] = df[f"tree_obs"] - df[f"tree_pred"]
        df[f"bark_resid"] = df[f"bark_obs"] - df[f"bark_pred"]
        df[f"branch_resid"] = df[f"branch_obs"] - df[f"branch_pred"]
        df[f"foliage_resid"] = df[f"foliage_obs"] - df[f"foliage_pred"]
        df[f"wood_resid"] = df[f"wood_obs"] - df[f"wood_pred"]
    else:
        raise Exception(f"Target: {cfg['rf_target']} is not supported")

    return df


def set_matching_axes(df, ax, x, y, resid_plot=False, buffer=5):
    all_vals = pd.concat((df[x], df[y]))
    ax.set_xlim([0, all_vals.max() + buffer])
    if resid_plot:
        y_abs = abs(df[y])
        y_max = max(y_abs) + buffer
        y_min = -abs(y_max)
        ax.set_ylim([y_min, y_max])
    else:
        ax.set_ylim([all_vals.min(), all_vals.max() + buffer])

    # Ensure x-axis does not become negative
    ax.set_xlim(left=0.)


def config_subplot_axis(df, metrics_df, comp, ax, x_axis, y_axis, resid_plot=False):
    x_vals = df[f"{comp}_{x_axis}"]
    y_vals = df[f"{comp}_{y_axis}"]
    ax.scatter(x_vals, y_vals, alpha=0.8, edgecolors='none', s=30)

    set_matching_axes(df, ax=ax, x=f"{comp}_obs", y=f"{comp}_{y_axis}", resid_plot=resid_plot)
    ax.text(0.1, 0.9,
            f"R2: {metrics_df.loc[f'{comp}', 'r2']}\nRMSE: {metrics_df.loc[f'{comp}', 'rmse']}",
            horizontalalignment='left', verticalalignment='top', transform=ax.transAxes)
    ax.title.set_text(comp.capitalize())


def get_metrics_make_plots(cfg, df, axis_lab, generate_plots=True):
    # Calculate test metrics for each component ------------------------------------------------------------------------

    # Create a data frame to store component metrics
    metrics_df = pd.DataFrame(columns=["r2", "rmse"],
                              index=["wood", "bark", "branch", "foliage",
                                     "tree"])

    if cfg['rf_target'] == 'biomass_comps':
        comp_list = metrics_df.index.tolist()
    elif cfg['rf_target'] == "total_agb":
        comp_list = ["tree"]
    else:
        raise Exception(f"Target variable: {cfg['rf_target']} is not supported")

    # Loop through each biomass component get model performance metrics
    for comp in comp_list:
        metrics_df.loc[comp, "r2"] = round(metrics.r2_score(y_true=df[f"{comp}_obs"], y_pred=df[f"{comp}_pred"]), 4)
        metrics_df.loc[comp, "rmse"] = round(
            sqrt(metrics.mean_squared_error(y_true=df[f"{comp}_obs"], y_pred=df[f"{comp}_pred"])), 4)

    if generate_plots:

        # Plot total AGB biomass obs. vs. predicted  -----------------------------------------------------------------------

        # Create plot
        fig = plt.figure(figsize=(10, 10))
        ax = fig.add_subplot(1, 1, 1)
        ax.scatter(df[f"tree_obs"], df[f"tree_pred"],
                   alpha=0.8, edgecolors='none', s=30)

        # Set axis labels
        ax.set_xlabel("Observed Tree AGB (Mg/ha)")
        ax.set_ylabel("Predicted Tree AGB (Mg/ha)")

        plt.figtext(0.05, 0.9,
                    f"R2: {metrics_df.loc['tree', 'r2']}\nRMSE: {metrics_df.loc['tree', 'rmse']}",
                    horizontalalignment="left",
                    verticalalignment="center",
                    transform=ax.transAxes)

        # Add title
        plt.title("Total Tree AGB Observed vs Predicted", fontdict=None, loc='center', fontsize=15)

        set_matching_axes(df, ax, x=f"tree_obs", y=f"tree_pred")

        ax.plot(ax.get_xlim(), ax.get_ylim(), ls="--", c=".3")

        plt.show()

        # Make residuals vs. fitted values plot for total AGB --------------------------------------------------------------
        fig = plt.figure(figsize=(10, 10))
        ax = fig.add_subplot(1, 1, 1)
        ax.scatter(df[f"tree_pred"], df[f"tree_resid"],
                   alpha=0.8, edgecolors='none', s=30)

        plt.axhline(y=0, color='black', linestyle='--')

        # Add title
        plt.title("Total Tree AGB Residuals", fontdict=None, loc='center', fontsize=15)

        set_matching_axes(df, ax, x=f"tree_obs", y=f"tree_resid", resid_plot=True)

        # Set axis labels
        ax.set_xlabel("Observed Tree AGB (Mg/ha)")
        ax.set_ylabel("Residuals Tree AGB (Mg/ha)")

        if cfg['rf_target'] == "biomass_comps":
            # Make subplots fir biomass component obs. vs. predicted   ---------------------------------------------------------
            fig, ax = plt.subplots(2, 2, figsize=(10, 10))

            # Add the main title
            fig.suptitle("Component Biomass Observed vs Predicted", fontsize=15)

            config_subplot_axis(df, metrics_df, comp="bark", ax=ax[0, 0], x_axis="obs", y_axis="pred")
            config_subplot_axis(df, metrics_df, comp="branch", ax=ax[1, 0], x_axis="obs", y_axis="pred")
            config_subplot_axis(df, metrics_df, comp="foliage", ax=ax[0, 1], x_axis="obs", y_axis="pred")
            config_subplot_axis(df, metrics_df, comp="wood", ax=ax[1, 1], x_axis="obs", y_axis="pred")

            # Add axis labels
            for axis in ax.flat:
                axis.set(xlabel=f"Observed {axis_lab}", ylabel=f"Predicted {axis_lab}")
                axis.plot(axis.get_xlim(), axis.get_ylim(), ls="--", c=".3")

            # set the spacing between subplots
            plt.subplots_adjust(left=0.1,
                                bottom=0.1,
                                right=0.9,
                                top=0.9,
                                wspace=0.3,
                                hspace=0.3)

            plt.show()

            # Make subplots for component biomass residuals --------------------------------------------------------------------
            fig, ax = plt.subplots(2, 2, figsize=(10, 10))

            # Add the main title
            fig.suptitle("Component Biomass Residuals", fontsize=15)

            config_subplot_axis(df, metrics_df, comp="bark", ax=ax[0, 0], x_axis="pred", y_axis="resid",
                                resid_plot=True)
            config_subplot_axis(df, metrics_df, comp="branch", ax=ax[1, 0], x_axis="pred", y_axis="resid",
                                resid_plot=True)
            config_subplot_axis(df, metrics_df, comp="foliage", ax=ax[0, 1], x_axis="pred", y_axis="resid",
                                resid_plot=True)
            config_subplot_axis(df, metrics_df, comp="wood", ax=ax[1, 1], x_axis="pred", y_axis="resid",
                                resid_plot=True)

            # Add axis labels
            for axis in ax.flat:
                axis.set(xlabel=axis_lab, ylabel='Residuals')
                axis.axhline(y=0, c="black", linestyle='--')

            # set the spacing between subplots
            plt.subplots_adjust(left=0.1,
                                bottom=0.1,
                                right=0.9,
                                top=0.9,
                                wspace=0.3,
                                hspace=0.3)

            plt.show()

    return metrics_df


def test_rf(cfg,
            eval_data,
            y_vars,
            features=None,
            model=None,
            axis_lab="Biomass Mg/ha",
            model_fpath=None,
            hp_tuning_mode=False
            ):
    # If modelling component biomass, update z-score conversion values (means and sds for each comp)
    z_info = update_z_score_conversion_info(cfg['data_dir'])

    ##############################
    # LOAD MODEL IF PATH SPECIFIED
    ##############################
    if model is None:
        # Load model
        model = load(model_fpath)
        # Extract feature names
        features = model.feature_names_in_
        print(f"Loading model {model_fpath} and using input features \n{features}")
    else:
        if hp_tuning_mode is False:
            print(f"Testing rf model with the following features:\n{model.feature_names_in_}")

    ###############
    # TEST RF MODEL
    ###############

    # Use the forest's predict method on the test data
    pred = model.predict(eval_data[features])
    obs = eval_data[y_vars].to_numpy()

    # Adjust array shape for total AGB modelling
    if cfg['rf_target'] == "total_agb":
        pred = np.reshape(pred, (pred.shape[0], 1))

    # Reconvert from z-score to Mg/ha if modelling biomass comps
    if cfg['rf_target'] == "biomass_comps":
        pred = re_convert_to_Mg_ha(z_info, z_components_arr=pred)
        obs = re_convert_to_Mg_ha(z_info, z_components_arr=obs)

    #Compute overall R2 and RMSE
    overall_r2 = metrics.r2_score(y_true=obs, y_pred=pred)
    overall_rmse = sqrt(metrics.mean_squared_error(y_true=obs, y_pred=pred))

    # Join obs and pred arrays
    arr = np.concatenate((obs, pred), axis=1)

    # Convert to data frame
    plot_id = list(eval_data['PlotID'])
    df = convert_array_to_df(arr, plot_id, cfg)

    if hp_tuning_mode:
        generate_plots = False
    else:
        generate_plots = True

    metrics_df = get_metrics_make_plots(cfg=cfg, df=df, axis_lab=axis_lab, generate_plots=generate_plots)

    return metrics_df, overall_r2, overall_rmse
