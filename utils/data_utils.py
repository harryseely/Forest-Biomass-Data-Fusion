import open3d as o3d
from laspy import read
import pandas as pd
import numpy as np
import torch
import os

from utils.ocnn_custom_utils import CustomTransform


def update_z_score_conversion_info(data_dir):
    """
    Loads a csv of the reference data and gets the mean and sd for use in converting back from z score.
    Global config dictionary (cfg) is updated with these values for each biomass component for conversion during training.
    :param data_dir: path to data directory
    :return: updated global config dictionary.
    """

    # Get paths
    ref_data_path = os.path.join(data_dir, 'biomass_labels.csv')
    ref_data = pd.read_csv(ref_data_path)

    # Create dict to store z_info for z-score conversion
    z_info = dict()

    # Get mean and sd for each component and update the global cfg
    for comp in ['bark', 'branch', 'foliage', 'wood']:
        z_info[f'{comp}_Mg_ha_mn'] = np.mean(ref_data[f'{comp}_Mg_ha'])
        z_info[f'{comp}_Mg_ha_sd'] = np.std(ref_data[f'{comp}_Mg_ha'])

    return z_info


def convert_from_z_score(z_vals, sd, mean):
    """
    Converts z-score back to original value using mean and sd
    :param cfg: global config dict that contains the mean and sd values needed for conversion
    :param z_vals: z-score values to be converted
    :param sd: standard deviation of original data
    :param mean: mean of original data
    :return: input values converted to back to original units
    """

    # X = Z * standard_deviation + mean
    converted_val = z_vals * sd + mean

    return converted_val


def re_convert_to_Mg_ha(z_info: dict, z_components_arr):
    """
    Converts array of component z score value back to biomass value in Mg/ha
    ***IMPORTANT: array needs to enter function with columns as follows: bark, branch, foliage, wood

    :param z_info: dict that contains the mean and sd values for each component needed for conversion
    :param z_components_arr: input np array of 'branch', 'bark', 'foliage', 'wood' values (in z score format)
    :return: tensor -> input values converted to Mg/ha units, note that this tensor no longer has gradients and is only for calculating performance metrics
    """

    # Send tensor to cpu if needed
    if torch.is_tensor(z_components_arr):
        converted_arr = z_components_arr.detach().clone()
    else:
        converted_arr = z_components_arr

    # Re-convert z-score to original value for each component
    for col_number, comp in zip(range(0, 4), ['bark', 'branch', 'foliage', 'wood']):
        comp_z_vals = converted_arr[:, col_number]
        converted_arr[:, col_number] = convert_from_z_score(comp_z_vals, sd=z_info[f'{comp}_Mg_ha_sd'],
                                                            mean=z_info[f'{comp}_Mg_ha_mn'])

    return converted_arr


def normalize_i(intensity_vals):
    i_norm = ((intensity_vals - min(intensity_vals)) / (max(intensity_vals) - min(
        intensity_vals)))
    return i_norm


def estimate_point_normals(points, visualize=False):
    """
    Uses open3d to estimate point normals.
    The "normals" denote the x,y,z components of the surface normal vector at each point
    :param points: numpy array of point cloud where first 3 columns are x,y,z coordinates
    :param visualize: whether to visualize the point cloud with normals using open3d (for debugging)
    :return: Point cloud with estimated normals.
    """

    # Select coordinates
    xyz = points[:, 0:3]

    # Create an empty point cloud
    pcd = o3d.geometry.PointCloud()

    # Pass xyz coords to point cloud class
    pcd.points = o3d.utility.Vector3dVector(xyz)

    # Estimate normals
    pcd.estimate_normals()

    # Convert normals to numpy array
    normals = np.asarray(pcd.normals)

    # Bind normals to original point cloud
    points_w_normals = np.concatenate((points, normals), axis=1)

    if visualize:
        o3d.visualization.draw_geometries([pcd], point_show_normal=True)

    return points_w_normals


def read_las_to_np(las_fpath, normalize_intensity=True, use_ground_points=None, centralize_coords=True, compute_normals=False):
    """
    IMPORTANT: format of np array storing lidar data returned has the following format:
    N x C
    Where N is the numer of points and C is the number of columns
    The current columns included in the np array by index are:
    0 - X
    1 - Y
    2 - Z
    3 - Intensity (Normalized or Raw, depending on argument)
    4 - Return Number
    5 - Classification
    6 - Scan Angle Rank
    7 - Number of Returns

    *If compute_normals is True
    8 - x component of normals
    9 - y component of normals
    10 - z component of normals

    :param las_fpath: filepath to las file
    :param normalize_intensity: whether to normalize intensity values
    :param use_ground_points: height below which to remove points, specify as None to use no height filter
    :param centralize_coords: whether to make all coords relative to center of point cloud (center is 0,0,0)
    :param compute_normals: whether to compute normals for each point
    :return: point cloud numpy array
    """

    # Read LAS for given plot ID
    inFile = read(las_fpath)

    # Correct for difference in file naming for scan angle (some LAS files call it scan_angle)
    try:
        scan_angle = inFile.scan_angle_rank
    except AttributeError:
        try:
            scan_angle = inFile.scan_angle
        except AttributeError:
            raise Exception("Issue with scan angle name in LAS file...")

    # Get coords coordinates
    points = np.vstack([inFile.x,
                        inFile.y,
                        inFile.z,
                        inFile.intensity,
                        inFile.return_number,
                        inFile.classification,
                        scan_angle,
                        inFile.number_of_returns
                        ]).transpose()

    if use_ground_points:
        pass
    else:
        # Filter the array by dropping all rows with a value of 2 (ground point) in the Classification column (4th)
        points = points[points[:, 4] != 2]

    if centralize_coords:
        points[:, 0:3] = points[:, 0:3] - np.mean(points[:, 0:3], axis=0)

    # Normalize Intensity
    if normalize_intensity:
        points[:, 3] = normalize_i(points[:, 3])


    if compute_normals:
        points = estimate_point_normals(points)

    # Check for NANs and report
    if np.isnan(points).any():
        raise ValueError('NaN values in input point cloud!')
    return points


def load_octree_sample(points, cfg, idx, augment):
    """
    Loads a sample from a las file and converts it to octree format
    :param points: point cloud numpy array
    :param cfg: config dict
    :param idx: sample idx
    :param augment: whether to apply data augmentation to the sample point cloud
    :return: dictionary containing point cloud data in octree format with target
    """

    # Convert points to octree tensor ------------------------------------------------------------------------------

    # Features index from 3:8 are intensity, return number, classification, scan angle rank, number of returns
    features = points[:, 3:8] if cfg['ocnn_use_feats'] else None

    # Indeces from 8:11 are x, y, z components of normals
    normals = points[:, 8:11] if cfg['use_normals'] else None

    # Convert to double to get network to work
    sample = {'points': points[:, 0:3],  # XYZ
              'normals': normals,
              'features': features
              }

    # Convert point cloud to octree format
    transform = CustomTransform(depth=cfg['octree_depth'],
                                full_depth=cfg['full_depth'],
                                use_normals=cfg['use_normals'],
                                augment=augment)

    sample = transform(sample, idx=idx)

    return sample


def augment_spectral_sample(spectra_row, spec_sd, sd_scaling_factor=0.1):
    aug_spectra_row = spectra_row.copy()
    for band in list(spectra_row.columns):
        sd_scaled = sd_scaling_factor * spec_sd[band]
        band_noise = np.random.normal(loc=0.0, scale=sd_scaled, size=1)
        aug_spectra_row[band] = spectra_row[band] + band_noise

    return aug_spectra_row


if __name__ == "__main__":

    import yaml
    import seaborn as sns
    import matplotlib.pyplot as plt

    # Read config
    with open("config.yaml", "r") as yamlfile:
        cfg = yaml.load(yamlfile, Loader=yaml.FullLoader)

    #Test Z-score conversion -------------------------------------------------------------------------------------------

    # Load biomass data reference data -----------------------------------------------------------------------------
    ref_data_path = os.path.join(cfg['data_dir'], 'model_input_plot_biomass_data.csv')
    df = pd.read_csv(ref_data_path, sep=",", header=0)

    # Update mean and sd and save for z-score conversion
    z_info = update_z_score_conversion_info(cfg['data_dir'])

    #Get true val
    val_Mg_ha = df[["branch_z", "bark_z", "foliage_z", "wood_z"]].to_numpy()

    #Get z score val
    val_z_score = df[["branch_Mg_ha", "bark_Mg_ha", "foliage_Mg_ha", "wood_Mg_ha"]].to_numpy()

    # Convert z-score
    vals_converted = re_convert_to_Mg_ha(z_info, z_components_arr=val_z_score)

    for i, comp in enumerate(['branch', 'bark', 'foliage', 'wood']):
        sns.scatterplot(x=val_Mg_ha[:, i], y=vals_converted[:, i], label=f"{comp} Mg/ha")
        plt.show()


