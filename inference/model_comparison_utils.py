
def check_model_type(row):
    if row['spec_cnn'] and row['include_thermal'] and row['terrain_cnn']:
        return 'Full' # ----> Everything
    elif row['spec_cnn'] and row['include_thermal']  and row['terrain_cnn'] == False:
        return 'L+S+ST' # ----> No Terrain
    elif row['spec_cnn'] and row['include_thermal'] == False and row['terrain_cnn']:
        return 'L+T+S' # ----> No Surface Temperature
    elif row['spec_cnn'] and row['include_thermal'] == False and row['terrain_cnn'] == False:
        return 'L+S' # ----> Lidar + Spectral Only
    elif row['spec_cnn'] == False and row['terrain_cnn']:
        return 'L+T' # ----> Lidar + Topography Only
    else:
        return 'L'