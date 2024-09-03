import pandas as pd
import sklearn.metrics as metrics
from math import sqrt
from matplotlib import pyplot as plt

def view_and_save_fig(fig_out_dir, file_desc):
    fig_export = plt.gcf()
    plt.show()
    plot_filepath = os.path.join(fig_out_dir, f'{file_desc}.png')
    if os.path.isfile(plot_filepath):
        os.remove(plot_filepath)
    fig_export.savefig(plot_filepath)


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


def config_subplot_axis(df, target, comp, ax, x_axis, y_axis, resid_plot=False):
    x_vals = df[f"{comp}_{target}_{x_axis}"]
    y_vals = df[f"{comp}_{target}_{y_axis}"]
    ax.scatter(x_vals, y_vals, alpha=0.8, edgecolors='none', s=30)

    set_matching_axes(df, ax=ax, x=f"{comp}_{target}_obs", y=f"{comp}_{target}_{y_axis}", resid_plot=resid_plot)
    ax.title.set_text(comp.capitalize())

def get_metrics_make_plots(df, target_label, datetime, axis_lab, fig_out_dir, view_plots=True):

    # Calculate test metrics for each component ------------------------------------------------------------------------

    # Create a data frame to store component metrics
    metrics_df = pd.DataFrame(columns=["r2", "rmse", "mae"],
                              index=[f"wood_{target_label}", f"bark_{target_label}", f"branch_{target_label}", f"foliage_{target_label}",
                                     f"tree_{target_label}"])

    comp_list = metrics_df.index.tolist()


    # Loop through each biomass component get model performance metrics
    for comp in comp_list:

        #R2
        metrics_df.loc[comp, "r2"] = round(metrics.r2_score(y_true=df[f"{comp}_obs"], y_pred=df[f"{comp}_pred"]), 4)

        #RMSE
        metrics_df.loc[comp, "rmse"] = round(
            sqrt(metrics.mean_squared_error(y_true=df[f"{comp}_obs"], y_pred=df[f"{comp}_pred"])), 4)

        #MAE
        metrics_df.loc[comp, "mae"] = round(
            metrics.mean_absolute_error(y_true=df[f"{comp}_obs"], y_pred=df[f"{comp}_pred"]), 4)

    print(metrics_df)

    # Plot total AGB biomass obs. vs. predicted  -----------------------------------------------------------------------
    if view_plots:
        # Create plot
        fig = plt.figure(figsize=(10, 10))
        ax = fig.add_subplot(1, 1, 1)
        ax.scatter(df[f"tree_{target_label}_obs"], df[f"tree_{target_label}_pred"],
                   alpha=0.8, edgecolors='none', s=30)

        # Set axis labels
        ax.set_xlabel("Observed Tree AGB (Mg/ha)")
        ax.set_ylabel("Predicted Tree AGB (Mg/ha)")

        plt.figtext(0.05, 0.9,
                    f"R2: {metrics_df.loc[f'tree_{target_label}', 'r2']}\nRMSE: {metrics_df.loc[f'tree_{target_label}', 'rmse']}\nMAE: {str(round(metrics_df.loc[f'tree_{target_label}', 'mae'], 2))}",
                    horizontalalignment="left",
                    verticalalignment="center",
                    transform=ax.transAxes)

        # Add title
        plt.title("Total Tree AGB Observed vs Predicted", fontdict=None, loc='center', fontsize=15)

        set_matching_axes(df, ax, x=f"tree_{target_label}_obs", y=f"tree_{target_label}_pred")

        ax.plot(ax.get_xlim(), ax.get_ylim(), ls="--", c=".3")

        view_and_save_fig(fig_out_dir, file_desc=f"tree_obs_vs_pred_{datetime}")

        # Make residuals vs. fitted values plot for total AGB --------------------------------------------------------------
        fig = plt.figure(figsize=(10, 10))
        ax = fig.add_subplot(1, 1, 1)
        ax.scatter(df[f"tree_{target_label}_pred"], df[f"tree_{target_label}_resid"],
                   alpha=0.8, edgecolors='none', s=30)

        plt.axhline(y=0, color='black', linestyle='--')

        # Add title
        plt.title("Total Tree AGB Residuals", fontdict=None, loc='center', fontsize=15)

        set_matching_axes(df, ax, x=f"tree_{target_label}_obs", y=f"tree_{target_label}_resid", resid_plot=True)

        # Set axis labels
        ax.set_xlabel("Observed Tree AGB (Mg/ha)")
        ax.set_ylabel("Residuals Tree AGB (Mg/ha)")

        view_and_save_fig(fig_out_dir, file_desc=f'tree_{datetime}')

        # Make subplots fir biomass component obs. vs. predicted   ---------------------------------------------------------
        fig, ax = plt.subplots(2, 2, figsize=(10, 10))

        # Add the main title
        fig.suptitle("Component Biomass Observed vs Predicted", fontsize=15)

        config_subplot_axis(df, target_label, comp="bark", ax=ax[0, 0], x_axis="obs", y_axis="pred")
        config_subplot_axis(df, target_label, comp="branch", ax=ax[1, 0], x_axis="obs", y_axis="pred")
        config_subplot_axis(df, target_label, comp="foliage", ax=ax[0, 1], x_axis="obs", y_axis="pred")
        config_subplot_axis(df, target_label, comp="wood", ax=ax[1, 1], x_axis="obs", y_axis="pred")

        # Add axis labels
        for axis in ax.flat:
            axis.set(xlabel=f"Observed {axis_lab}", ylabel=f"Predicted {axis_lab}")
            axis.plot(axis.get_xlim(), axis.get_ylim(), ls="--", c=".3")

        # set the spacing between subplots
        plt.subplots_adjust(left=0.1,
                            bottom=0.1,
                            right=0.9,
                            top=0.9,
                            wspace=0.5,  # increased from 0.3
                            hspace=0.5)  # increased from 0.3

        view_and_save_fig(fig_out_dir, file_desc=f'component_obs_vs_pred_{datetime}')

        # Make subplots for component biomass residuals --------------------------------------------------------------------
        fig, ax = plt.subplots(2, 2, figsize=(10, 10))

        # Add the main title
        fig.suptitle("Component Biomass Residuals", fontsize=15)

        config_subplot_axis(df, target_label, comp="bark", ax=ax[0, 0], x_axis="pred", y_axis="resid",
                            resid_plot=True)
        config_subplot_axis(df, target_label, comp="branch", ax=ax[1, 0], x_axis="pred", y_axis="resid",
                            resid_plot=True)
        config_subplot_axis(df, target_label, comp="foliage", ax=ax[0, 1], x_axis="pred", y_axis="resid",
                            resid_plot=True)
        config_subplot_axis(df, target_label, comp="wood", ax=ax[1, 1], x_axis="pred", y_axis="resid",
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

        # Save plot
        view_and_save_fig(fig_out_dir, file_desc=f'component_residuals_{datetime}')

    return metrics_df