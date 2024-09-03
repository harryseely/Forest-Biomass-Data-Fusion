# Other Modules
import pprint as pp
import os
import yaml
import pandas as pd

# My scripts
from random_forest.train_rf import train_rf


def main(cfg):
    cfg['rf_target'] = "biomass_comps"
    cfg['logging'] = False

    # Random Forest Hyperparameters
    cfg['rf_model'] = True
    cfg['rf_var_impt_thresh'] = 1e-30  # Very low threshold to include all features
    cfg['rf_n_estimators'] = 200
    cfg['rf_max_depth'] = 90
    cfg['rf_min_samples_split'] = 12
    cfg['rf_min_samples_leaf'] = 4
    cfg['rf_max_features'] = 1
    cfg['rf_max_samples'] = 0.9

    # Report hyperparameters
    print("\nHyperparameters:\n")
    pp.pprint(cfg, width=1, sort_dicts=False)

    # Create model filepath where best model will be saved and updated
    model_filename = "rf_model.pt"  # f'rf_{dt.now().strftime("%Y_%m_%d_%H_%M")}.pt'
    cfg['rf_model_filepath'] = os.path.join("../rf_models", model_filename)

    # Set target var
    if cfg['z_score']:
        y_vars = ['foliage_z', 'branch_z', 'bark_z', 'wood_z']

    else:
        y_vars = ['foliage_Mg_ha', 'branch_Mg_ha', 'bark_Mg_ha', 'wood_Mg_ha']

    data_fpath = os.path.join(cfg['data_dir'], 'model_input_plot_biomass_data.csv')
    df = pd.read_csv(data_fpath)

    spectral_cols = df.columns[df.columns.str.contains('b_')].tolist()

    metrics_df, runtime, train_output, selected_feat_df = train_rf(cfg,
                                                                   y_vars,
                                                                   x_vars=spectral_cols,
                                                                   log=cfg['logging'],
                                                                   hp_tuning_mode=False)


if __name__ == '__main__':
    # Read config
    with open(r"../config.yaml", "r") as yamlfile:
        cfg = yaml.load(yamlfile, Loader=yaml.FullLoader)

    # Run it
    main(cfg)
