from tqdm import tqdm
import os
import pandas as pd
from sklearn.ensemble import RandomForestRegressor

def rf_band_screening(cfg, n_runs):
    # Set target var
    if cfg['z_score']:
        y_vars = ['foliage_z', 'branch_z', 'bark_z', 'wood_z']

    else:
        y_vars = ['foliage_Mg_ha', 'branch_Mg_ha', 'bark_Mg_ha', 'wood_Mg_ha']

    ref_data_path = os.path.join(cfg['data_dir'], cfg['dataset'], 'biomass_labels.csv')
    df = pd.read_csv(ref_data_path, sep=",", header=0)

    # Whether to include thermal
    if cfg['include_thermal'] is False:
        thermal_cols = df.columns[df.columns.str.contains('thermal')]
        df = df.drop(columns=thermal_cols)

    # Reduce spectral cols to target years only
    x_vars = df.columns[df.columns.str.contains('b_')].tolist()

    # Run random forest N times to get a good estimation of feature importance
    feat_importance_df = pd.DataFrame([{'feat': 'dummy', 'importance': 0}])

    for _ in tqdm(range(n_runs), desc="RF Band Screening", leave=False):
        # Define random forest classifier using default settings
        model = RandomForestRegressor(n_jobs=-1, verbose=0)

        # Train model to get initial assessment of feature importance
        model.fit(X=df[x_vars], y=df[y_vars])

        # Create df of feature importance
        importance_df_n = pd.DataFrame({'feat': list(model.feature_names_in_),
                                        'importance': model.feature_importances_})

        # Append to main df
        feat_importance_df = pd.concat([feat_importance_df, importance_df_n], axis=0)

        # Sort df based on feature importance
        feat_importance_df = feat_importance_df.sort_values(by='importance', ascending=False).reset_index(drop=True)

    # Summarize feature importance by mean and sd as seperate columns, and name columns appropriately
    feat_importance_df = (feat_importance_df.
                          groupby('feat').
                          agg({'importance': ['mean', 'std']}).
                          reset_index()
                          )

    # Fix column names
    feat_importance_df.columns = ['feature', 'importance_mn', 'importance_sd']

    # Drop dummy column
    feat_importance_df = feat_importance_df[feat_importance_df['feature'] != 'dummy']

    #Ensure bands are in correct order if some are are removed
    band_order_df = pd.DataFrame({'feature': x_vars})
    band_order_df['order'] = band_order_df.index

    #Join to feature importance df
    feat_importance_df = pd.merge(band_order_df, feat_importance_df, on='feature', how='left')

    #Sort by importance and keep the top k features
    feat_importance_df = feat_importance_df.sort_values(by='importance_mn', ascending=False)

    #Only keep the top k rows with the highest importance
    feat_importance_df = feat_importance_df.head(cfg['rf_top_n_feats'])

    #Ensure bands are in original order regardless of importance score
    feat_importance_df = feat_importance_df.sort_values(by='order')

    return feat_importance_df


def band_selection(cfg, n_runs=10):

    # Random forest band pre-screening
    if cfg['bands'] == 'use_rf_screening':
        band_importance_df = rf_band_screening(cfg, n_runs)
        selected_bands = band_importance_df['feature'].tolist()
        selected_bands = ",".join(selected_bands)

        print(f"RF selected bands:\n{selected_bands}")

    elif cfg['bands'] == 'all':
        # Load biomass data reference data to get all band names
        ref_data_path = os.path.join(cfg['data_dir'], cfg['dataset'], 'biomass_labels.csv')
        df = pd.read_csv(ref_data_path, sep=",", header=0)
        selected_bands = df.columns[df.columns.str.contains('b_')].tolist()
        selected_bands = ",".join(selected_bands)

        print(f"No RF band screening.")

    else:
        # Selected bands are already specified in the config in a comma separated string
        selected_bands = cfg['bands']

    # Convert to comma seperated string and store in config
    cfg['bands'] = selected_bands


    return cfg


if __name__ == '__main__':

    import yaml

    # Read config
    with open(r"../config.yaml", "r") as yamlfile:
        cfg = yaml.load(yamlfile, Loader=yaml.FullLoader)

    cfg['bands'] = 'use_rf_screening'

    # Run it
    cfg = band_selection(cfg, n_runs=10)

