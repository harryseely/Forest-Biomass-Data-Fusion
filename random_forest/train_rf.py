# Modules
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from time import time
import os
import wandb

# My scripts
from random_forest.test_rf import test_rf
from utils.training_utils import name_run

def train_rf(cfg,
             y_vars,
             x_vars=(),
             log=False,
             hp_tuning_mode=False,
             test_rf = True,
             verbose=True
             ):
    t0 = time()


    if log:

        run_name = name_run(cfg)

        # More info about logging: https://lightning.ai/docs/pytorch/stable/api/pytorch_lightning.loggers.wandb.html
        # Best practices for wandb: https://wandb.ai/wandb/pytorch-lightning-e2e/reports/W-B-Best-Practices-Guide--VmlldzozNTU1ODY1
        with open(cfg['wandb_key']) as f:
            wandb_key = f.readlines()[0]

        wandb.login(key=wandb_key)

        # Integrate wandb logger with lightning
        logger = wandb.init(
            name=run_name,
            config=cfg,
            id=run_name,
            project="RQ2",
            save_code=True,
            mode="online",
            allow_val_change=True,
            job_type='training',  # for organizing runs (e.g. preprocessing vs. training)
            resume="allow",
        )
    else:
        logger = None

    # Load train and test datasets
    data_fpath = os.path.join(cfg['data_dir'], 'model_input_plot_biomass_data.csv')
    df = pd.read_csv(data_fpath)

    # Select use cols from df
    use_cols = y_vars + x_vars
    use_cols.append("PlotID")

    #Select train/val/test split based on fold column
    fold_col = f"fold_{cfg['data_fold']}"
    train_data = df[df[fold_col] == 'train']
    val_data = df[df[fold_col] == 'val']
    test_data = df[df[fold_col] == 'test']

    #Reduce dfs to use columns
    train_data = train_data[use_cols]
    val_data = val_data[use_cols]
    test_data = test_data[use_cols]

    ####################
    # VARIABLE SELECTION
    ####################
    if (hp_tuning_mode is False) & (logger is not None):
        print(f"\n    Performing RF Variable Selection")

    # Specify y_vars and input features and ensure these and other cols are excluded from input predictors
    excluded_cols = y_vars + ['PlotID']
    features = [col for col in train_data.columns if col not in excluded_cols]

    # Define random forest classifier using default settings
    model = RandomForestRegressor(n_jobs=-1, verbose=0)

    # Train model to get initial assessment of feature importance
    model.fit(X=train_data[features], y=train_data[y_vars])

    # Create df of feature importance
    feat_importance_df = pd.DataFrame({'feat': list(model.feature_names_in_),
                                       'importance': model.feature_importances_})

    # Sort df based on feature importance
    feat_importance_df = feat_importance_df.sort_values(by='importance', ascending=False).reset_index(drop=True)

    # Subset to features that have an importance score greater than specified threshold
    selected_feat_df = feat_importance_df.loc[lambda x: x['importance'] > cfg['rf_var_impt_thresh']]

    if logger is not None:
        # Log feature importance
        logger.log({"feature_importance": wandb.Table(data=feat_importance_df)})

    # Get list of selected features
    features = selected_feat_df['feat'].tolist()

    ##################
    # TRAIN RF MODEL
    ##################

    # Create a new random forest object
    model = RandomForestRegressor(
        n_estimators=cfg['rf_n_estimators'],
        max_depth=cfg['rf_max_depth'],
        min_samples_split=cfg['rf_min_samples_split'],
        min_samples_leaf=cfg['rf_min_samples_leaf'],
        max_features=cfg['rf_max_features'],
        max_samples=cfg['rf_max_samples'],
        random_state=66,
        verbose=0,
        n_jobs=-1,
    )

    # fit the regressor with x and y data
    model.fit(X=train_data[features], y=train_data[y_vars])

    # Add feature names and target as an attributes to model object
    model.feature_names = features

    # Evaluate model using test dataset (use val dataset when HP tuning)
    if hp_tuning_mode:
        eval_data = val_data
    else:
        eval_data = test_data

    if test_rf:
        metrics_df, overall_r2, overall_rmse = test_rf(cfg,
                                                       eval_data,
                                                       y_vars=y_vars,
                                                       features=features,
                                                       model=model,
                                                       hp_tuning_mode=hp_tuning_mode
                                                       )
    else:
        metrics_df = None
        overall_r2 = None
        overall_rmse = None

    if hp_tuning_mode is False:
        # Record runtime
        end_time = time()
        runtime = round((end_time - t0) / 60, 4)
        print(f"RF training time: {round(runtime * 60, 4)} seconds ({runtime} minutes)")

        train_output = {
            'n_train': len(train_data),
            'n_test': len(test_data),
            'n_val': len(val_data),
            'features': features,
        }

        # Log metrics and train output
        if logger is not None:
            metrics_df['index'] = metrics_df.index
            metrics_df.reset_index(inplace=True)
            metrics_df.rename(columns={'index': 'component'}, inplace=True)

            # Convert df to dict for logging
            metrics_df_long = metrics_df.melt(id_vars=['component'], value_vars=['r2', 'rmse'])
            metrics_df_long['comp_metric'] = metrics_df_long["component"] + "_" + metrics_df_long["variable"]
            metrics_dict = metrics_df_long.set_index('comp_metric')['value'].to_dict()

            # Add overall metrics
            metrics_dict['overall_r2'] = overall_r2
            metrics_dict['overall_rmse'] = overall_rmse

            # Log training details and test metrics
            logger.log(train_output)
            logger.log(metrics_dict)

        return metrics_df, runtime, train_output, selected_feat_df

    else:
        return overall_rmse
