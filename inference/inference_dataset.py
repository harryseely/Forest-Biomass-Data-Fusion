# Other packages
import os.path
import pytorch_lightning as pl
from torch.utils.data import DataLoader, Dataset
import numpy as np
import torch
import cupy as cp
import rasterio
import geopandas as gpd
from shapely.geometry import Polygon
from tqdm import tqdm
import laspy
import pandas as pd
import matplotlib.pyplot as plt
from rasterio.windows import from_bounds

# My base
from utils.ocnn_custom_utils import CustomCollateBatch
from utils.data_utils import load_octree_sample, read_las_to_np


def generate_bboxes(x_min, x_max, y_min, y_max, cell_size, out_shp_fpath=None):
    """
    Generates a list of bounding boxes for a given area and cell size.
    Outputs an array of bboxes is in the form [[[x_min, x_max], [y_min, y_max]], ... n-bboxes]

    :param x_min:
    :param x_max:
    :param y_min:
    :param y_max:
    :param cell_size:
    :param out_shp_fpath: filepath to save the bboxes to a shapefile
    :return:
    """

    # Determine the number of cells in each direction
    x_cells = int((x_max - x_min) / cell_size)
    y_cells = int((y_max - y_min) / cell_size)

    # Empty array to store bboxes
    bboxes = np.empty((x_cells * y_cells, 2, 2))

    # Loop through all cells in the grid and create a bbox for each
    for i in tqdm(range(x_cells), desc=f"Generating grid of {x_cells * y_cells} bounding boxes..."):
        for j in range(y_cells):
            bbox = np.array(((x_min + i * cell_size, x_min + (i + 1) * cell_size),
                             (y_min + j * cell_size, y_min + (j + 1) * cell_size)))

            bboxes[i * y_cells + j] = bbox

    # Save to shapefile
    if out_shp_fpath is not None:
        # Create a list of Polygon objects from the bboxes
        polygons = [Polygon(
            [(bbox[0][0], bbox[1][0]), (bbox[0][0], bbox[1][1]), (bbox[0][1], bbox[1][1]), (bbox[0][1], bbox[1][0])])
            for
            bbox in bboxes]

        # Convert the list of Polygon objects into a GeoSeries
        geoseries = gpd.GeoSeries(polygons)

        # Create a GeoDataFrame from the GeoSeries
        bboxes_gdf = gpd.GeoDataFrame(geometry=geoseries)

        # Export bounding boxes to shapefile
        bboxes_gdf.to_file(out_shp_fpath)

        print(f"Bounding boxes saved to:/n{out_shp_fpath}/n")

    return bboxes


def get_las_bbox(las_fpaths, progress_bar=True):
    """"
    Returns a bounding box for a list of las/laz files

    :param las_fpaths: list of las/laz file paths
    :param progress_bar: bool, default True, show progress bar
    :return: las bounding box with shape [xmin, ymin, xmax, ymax]
    """

    xmins = []
    ymins = []
    xmaxs = []
    ymaxs = []

    for las_fpath in tqdm(las_fpaths, disable=not progress_bar,
                          desc="Checking las file headers to determine bounding box..."):
        with laspy.open(las_fpath) as f:
            header = f.header

            xmins.append(header.mins[0])
            ymins.append(header.mins[1])
            xmaxs.append(header.maxs[0])
            ymaxs.append(header.maxs[1])

    xmin = min(xmins)
    ymin = min(ymins)
    xmax = max(xmaxs)
    ymax = max(ymaxs)

    bbox = [xmin, ymin, xmax, ymax]

    return bbox


def clip_las_to_bboxes(las, bboxes, device_id, progress_bar=False):
    """
    Uses cupy to clip las points to bounding boxes by looping through each bbox and extracting the intersecting las points

    :param las: input numpy array of las points with shape (n_points, n_cols)
    :param bboxes: numpy array of bounding box coordinates with shape [[[x_min, x_max], [y_min, y_max]], ... n-bboxes]
    :param device_id: Integer indicating GPU device id (e.g., 0) to use for cupy operations
    :param progress_bar: Boolean to indicate whether to display a progress bar
    :return: list of las points clipped to each bounding box of length equal to the number of bboxes intersecting with the las tile
    """

    # Ensure all cupy operations are done on the target GPU (as specified under device_id)
    with cp.cuda.Device(device_id):

        # Create empty array to store las tiles
        las_clipped_ls = []
        bbox_centers = []
        bbox_save_ls = []

        # Get a bbox for the las tile
        min_x, min_y = np.nanmin(las[:, 0]), np.nanmin(las[:, 1])
        max_x, max_y = np.nanmax(las[:, 0]), np.nanmax(las[:, 1])

        las_bbox = np.asarray([[min_x, max_x], [min_y, max_y]])

        # Create mask to filter out bboxes that do not intersect with the las tile
        mask = (bboxes[:, 0, 0] >= las_bbox[0][0]) & (bboxes[:, 0, 1] <= las_bbox[0][1]) & \
               (bboxes[:, 1, 0] >= las_bbox[1][0]) & (bboxes[:, 1, 1] <= las_bbox[1][1])

        bboxes_in_las = bboxes[mask]

        # Send las and bboxes to GPU to apply cupy operations
        las_gpu = cp.asarray(las)
        bboxes_in_las_gpu = cp.asarray(bboxes_in_las)

        # Iterate through bounding boxes to extract intersecting las points
        for i in tqdm(range(bboxes_in_las.shape[0]), desc=f"GPU {device_id}: Clipping bboxes...", colour='magenta',
                      disable=not progress_bar):

            try:

                # Create mask based on bbox
                las_mask = ((las_gpu[:, 0] >= bboxes_in_las_gpu[i, 0, 0]) & (
                            las_gpu[:, 0] <= bboxes_in_las_gpu[i, 0, 1]) & (
                                    las_gpu[:, 1] >= bboxes_in_las_gpu[i, 1, 0]) & (
                                        las_gpu[:, 1] <= bboxes_in_las_gpu[i, 1, 1]))

                # Extract the points contained within the bbox
                las_clipped = las_gpu[las_mask]

                # Centralize the coordinates
                las_clipped[:, 0:3] = las_clipped[:, 0:3] - np.mean(las_clipped[:, 0:3], axis=0)

                # Send array back to CPU
                las_clipped_cpu = cp.asnumpy(las_clipped)

                # Add las points to list
                las_clipped_ls.append(las_clipped_cpu)

                # Record the bbox
                bbox_save_ls.append(bboxes_in_las[i])

                # Record the center location of each bbox for rasterization later and convert coords to tuple
                bbox_centers.append(tuple(bboxes_in_las[i].mean(axis=1)))


            except Exception as e:
                print(f"Error clipping las to bbox:/n{e}")

        # Clear the GPU
        del las_gpu

        # Hold CPU process until all GPU cupy processes are complete
        cp.cuda.Stream.null.synchronize()

    return las_clipped_ls, bbox_save_ls, bbox_centers


def clip_topo_ras(bbox, topo_ras_fpath, topo_ras_dims, buf_d, plot_topo_ras=False, verbose=False):
    """
    Extracts a topo raster for a given bbox. The relative elevation is calculated for the clipped raster by subtracting the elevation at the center pixel from all pixels in the raster.

    :param bbox: bboxe with format [[[x_min, x_max], [y_min, y_max]], ...]
    :param topo_ras_fpath: file path to the topo raster
    :param topo_ras_dims: required topo raster dimensions (x pixels, y pixels)
    :param buf_d: buffer distance for bbox to get correct local topo raster
    :param plot_topo_ras: whether to visualize the clipped topo raster
    :param verbose: whether to print additional information

    :return: Extracted topo raster with relative elevation in numpy format
    """
    try:
        #Read topo in raster within bbox window
        with rasterio.open(topo_ras_fpath) as src:
            # Define the bounding box with buffer
            minx = bbox[0, 0] - buf_d
            maxx = bbox[0, 1] + buf_d
            miny = bbox[1, 0] - buf_d
            maxy = bbox[1, 1] + buf_d

            # Create a window using the bounding box
            window = from_bounds(minx, miny, maxx, maxy, src.transform)

            # Read the window from the raster
            ras_clip = src.read(window=window)

        #Check if the clipped raster has the correct dimensions
        if not ras_clip.shape == topo_ras_dims:
            if verbose:
                print(f"Topographic raster dims are incorrect./nExpected: {topo_ras_dims}, got: {ras_clip.shape[1:3]}/n")
                print("Cropping raster to correct dimensions.")

            ras_clip = ras_clip[:, :topo_ras_dims[0], :topo_ras_dims[1]]

        #Get the value of approximate center pixel elevation value
        center_elev = ras_clip[0, int(topo_ras_dims[0] / 2), int(topo_ras_dims[1] / 2)]

        #Add relative elevation band
        elev_rel = ras_clip[0, :, :] - center_elev

        #Combine topo ras with relative elevation
        ras_clip = np.concatenate([ras_clip, elev_rel[np.newaxis, :, :]], axis=0)

        if plot_topo_ras:
            print("Plotting topo raster.")
            #Plot a grid of the four bands in the topo raster
            fig, axs = plt.subplots(2, 2, figsize=(10, 10))
            for i in range(4):
                ax = axs[int(i / 2), i % 2]
                ax.imshow(ras_clip[i], cmap='viridis')
                ax.set_title(f"Band {i}")

    except Exception as e:
        print(f"Error clipping topographic raster to bbox: {bbox}/nError msg:/n{e}")
        ras_clip = None

    return ras_clip



class InferenceDataset(Dataset):
    def __init__(self, cfg, bboxes, las_fpath, device_id, topo_ras_buf_d, topo_ras_dims, topo_ras_fpath, landsat_fpath):

        """
        :param cfg: config file
        :param bboxes: numpy array of bounding box coordinates with shape [[[x_min, x_max], [y_min, y_max]], ... n-bboxes]
        :param las_fpath: filepath to the lidar file (LAS/LAZ)
        :param device_id: Integer indicating GPU device id (e.g., 0) to use for cupy operations
        :param topo_ras_buf_d: buffer distance for bbox to get correct local topo raster
        :param topo_ras_dims: required topo raster dimensions (x pixels, y pixels)
        :param topo_ras_fpath: filepath to the topographic raster
        :param landsat_fpath: filepath to the landsat image

        Dataset used for inference with octree-CNN + 1-D CNN

        The current draft version of this Class is to do the following steps:

        Init method:
         1) load a single LAS tile
         2) loop through every bounding box in the grid and extract las points if within a given bbox
         3) extract the topographic raster for each bbox (if using 2-D CNN)
         4) extract spectral signatures for each bbox centre coordinate (if using 1-D CNN)

        Getitem method:
            1) load the octree sample for each bbox
            2) load the topo raster for each bbox (if using 2-D CNN)
            3) load the spectral signature for each bbox (if using 1-D CNN)
            4) return a dict containing the sample with selected data modalities

        """

        # Attach cfg ---------------------------------------------------------------------------------------------------
        self.cfg = cfg

        # Read the las file into a numpy array
        las = read_las_to_np(las_fpath, use_ground_points=cfg['use_ground_points'], centralize_coords=False,
                             normalize_intensity=True, compute_normals=cfg['use_normals'])

        # Clip the las points to the bounding boxes
        print(f"Clipping {os.path.basename(las_fpath)} to bounding boxes...")

        self.las_clipped_ls, self.bboxes, self.bbox_centers = clip_las_to_bboxes(las=las, bboxes=bboxes, device_id=device_id)

        #Clip the topo raster to the bounding boxes
        if self.cfg['terrain_cnn']:

            self.topo_ras_clipped_ls = []
            for bbox_i in tqdm(self.bboxes, desc=f"Clipping topographic raster to bounding boxes for {os.path.basename(las_fpath)}"):
                self.topo_ras_clipped_ls.append(clip_topo_ras(bbox_i,
                                                  topo_ras_fpath=topo_ras_fpath,
                                                  topo_ras_dims=topo_ras_dims,
                                                  buf_d=topo_ras_buf_d))

            # Ensure that each point cloud has a corresponding topo raster
            assert len(self.las_clipped_ls) == len(
                self.topo_ras_clipped_ls), "Number of las tiles and topo rasters do not match!"

        # Read the Landsat imagery and extract the spectral values for each bbox center coordinate
        if self.cfg['spec_cnn']:

            with rasterio.open(landsat_fpath) as src:

                # Sample the raster at the given coordinates
                spectral_vals = src.sample(self.bbox_centers)

                spectral_vals_list = list(spectral_vals)

                # Scale the thermal values to match other bands
                spectral_vals_list = [np.append(sv[:7], (sv[7] / 1000)) for sv in spectral_vals_list]

                # reshape to 1D array
                self.spectral_vals_list = [sv.reshape(1, -1) for sv in spectral_vals_list]

            # Ensure that each point cloud has a corresponding spectral signature
            assert len(self.las_clipped_ls) == len(
                self.spectral_vals_list), "Number of las tiles and spectral values do not match!"



        super().__init__()

    def __len__(self):
        return len(self.las_clipped_ls)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        # Create dict to store all the pixel data
        pixel_dict = dict()

        pixel_dict['contains_points'] = True

        # Add the bbox centre coordinate to the dict
        pixel_dict['bbox_center'] = self.bbox_centers[idx]

        # Select point clouds
        pc = self.las_clipped_ls[idx]

        if pc.shape[0] == 0:
            print(f"Bounding box with center at ({self.bbox_centers[idx]}) contains zero points.")

            pixel_dict['contains_points'] = False

            #If pc is empty, create a dummy pc as a placeholder with 1 point
            dummy_pc = np.ones([1, 11])

            # Load octree sample from dummy point cloud
            octree_dict = load_octree_sample(dummy_pc, cfg=self.cfg, idx=idx, augment=False)

        else:
            # Load octree sample from point cloud
            octree_dict = load_octree_sample(pc, cfg=self.cfg, idx=idx, augment=False)

        # Add octree info to sample
        pixel_dict.update(octree_dict)

        #Add topo raster to sample
        if self.cfg['terrain_cnn']:
            pixel_dict['terrain'] = torch.from_numpy(self.topo_ras_clipped_ls[idx]).float()

        # Add spectral data to sample
        if self.cfg['spec_cnn']:
            pixel_dict['spectral'] = torch.from_numpy(self.spectral_vals_list[idx]).float()

        return pixel_dict


class InferenceDataModule(pl.LightningDataModule):
    def __init__(self, cfg, bboxes, las_fpath, topo_ras_buf_d, topo_ras_dims, topo_ras_fpath, landsat_fpath, device_id):
        super().__init__()
        self.cfg = cfg
        self.bboxes = bboxes
        self.las_fpath = las_fpath
        self.topo_ras_buf_d = topo_ras_buf_d
        self.topo_ras_dims = topo_ras_dims
        self.topo_ras_fpath = topo_ras_fpath
        self.landsat_fpath = landsat_fpath
        self.device_id = device_id

        self.collate_fn = CustomCollateBatch(batch_size=cfg['batch_size'])

    def setup(self, stage: str):
        # Currently predict stage is only set up to use test dataset
        if stage == "predict":
            self.predict_dataset = InferenceDataset(cfg=self.cfg,
                                                    bboxes=self.bboxes,
                                                    las_fpath=self.las_fpath,
                                                    device_id=self.device_id,
                                                    topo_ras_buf_d = self.topo_ras_buf_d,
                                                    topo_ras_dims=self.topo_ras_dims,
                                                    topo_ras_fpath=self.topo_ras_fpath,
                                                    landsat_fpath=self.landsat_fpath,
                                                    )

    def predict_dataloader(self):
        return DataLoader(self.predict_dataset,
                          batch_size=self.cfg['batch_size'],
                          pin_memory=True,
                          shuffle=False,
                          collate_fn=self.collate_fn,
                          num_workers=self.cfg['n_workers'])


if __name__ == "__main__":

    import yaml

    # General params
    batch_size = 512
    ras_res = 22.56
    clean_las_dir = r"E:/NB_Gov_Lidar_Tiles_Clean"
    bbox_shp_out_fpath = None  # r"D:/Sync/RQ2/Analysis/data/inference_data/inference_bboxes.shp"
    lidar_index_fpath = r"D:/Sync/RQ2/Analysis/data/inference_data/inference_cleaned_lidar_index.shp"
    base_ras_dir = r'E:/rq2_pred_biomass_rasters'
    crs = "EPSG:2953"
    all_clean_las_fpaths = pd.read_csv("D:/Sync/RQ2/Analysis/data/inference_data/inf_tile_names.csv")[
        'filename'].tolist()


    # Get the lidar data bounding box coordinates (need to use all cleaned lidar files to build complete grid)
    lidar_bounds = get_las_bbox(all_clean_las_fpaths)

    bboxes = generate_bboxes(x_min=lidar_bounds[0],
                             x_max=lidar_bounds[2],
                             y_min=lidar_bounds[1],
                             y_max=lidar_bounds[3],
                             cell_size=ras_res,
                             out_shp_fpath=bbox_shp_out_fpath
                             )

    # Set buffer distance (meters) to get correct topo raster dims
    topo_ras_buf_d = 2

    # Read config
    with open("D:/Sync/RQ2/Analysis/config.yaml", "r") as yamlfile:
        cfg = yaml.load(yamlfile, Loader=yaml.FullLoader)

    data_module = InferenceDataModule(cfg=cfg,
                     bboxes=bboxes,
                     las_fpath=all_clean_las_fpaths[1],
                     device_id=1,
                     topo_ras_buf_d=2,
                     topo_ras_dims=(14, 14),
                     topo_ras_fpath=r"D:/Sync/RQ2/Analysis/data/inference_data/inference_topo_ras.tif",
                     landsat_fpath = r"D:/Sync/RQ2/Analysis/data/new_brunswick/landsat_imagery/landsat_8_2018.tif"
                                      )


    #Test initalizing dataset
    data_module.setup("predict")

    print("Inference dataset created successfully.")

