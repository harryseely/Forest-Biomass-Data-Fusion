import os
import yaml
import torch
from itertools import chain
import time
from geocube.api.core import make_geocube
from geocube.rasterize import rasterize_points_griddata
import pandas as pd
from pathlib import Path
import multiprocessing as mp
from tqdm import tqdm

from utils.trainer import LitModel
from utils.test_model import model_predict
from model_comparison_utils import check_model_type
from inference.inference_dataset import InferenceDataModule, generate_bboxes, get_las_bbox





def rasterize_gridded_points(centroids, out_ras_fpath, col_names, out_resolution, record_time=False):
    """

    IMPORTANT: this function assumes input centroids are on an even grid!

    Uses the geocube package to convert a gridded point dataset in the form of a geaodataframe to a rioxarray and then
    exports this to geotiff.
    :param centroids: geopandas dataframe of evenly spaced grid centroids
    :param out_ras_fpath: output raster file path for exported geotiff
    :param col_names: list of column names in centroids gdf to rasterize
    :param out_resolution: output resolution of raster in units of the CRS
    :param record_time: whether to report the runtime of the function
    :return:
    """

    if record_time:
        # Record start time
        start_time = time.time()

    # Rasterize Points to Gridded Xarray Dataset
    geo_grid = make_geocube(
        vector_data=centroids,
        measurements=col_names,
        resolution=out_resolution,
        output_crs=centroids.crs,
        rasterize_function=rasterize_points_griddata,
        fill=-9999,  # NA value
    )

    print("Rasterized points to xarray dataset")

    # Export to GeoTiff
    geo_grid.rio.to_raster(out_ras_fpath)

    print(f"Exported raster to {out_ras_fpath}")

    if record_time:
        # Record runtime
        end_time = time.time()
        runtime_seconds = round((end_time - start_time), 2)

        print(f"Finished rasterizing. Time required:\n{runtime_seconds} seconds.")





def predict_to_raster(lightning_model, ckpt_file, cfg, las_fpath, device_id, checkpoint_dir, crs, bboxes, topo_ras_buf_d, topo_ras_dims, topo_ras_fpath, landsat_ras_fpath,
                      out_ras_dir, pred_ras_res, progress_bar=False, comp_names=('bark', 'branch', 'foliage', 'wood')):
    """
    Loads a lidar tile, clips tile to bounding boxes representing pixels in a raster, and generates predictions for
    each pixel. Predictions are saved to a raster file.

    :param lightning_model: pytorch lightning model object
    :param ckpt_file: absolute filepath to model checkpoint
    :param cfg: dictionary containing config parameters
    :param las_fpath: absolute filepath to input lidar tile
    :param device_id: Integer indicating GPU device to use (e.g., 0). Default is 0.
    :param checkpoint_dir: Directory where config and model checkpoint are stored
    :param crs: coordinate reference system of input lidar tile and generated raster
    :param bboxes: array of bboxes for each pixel in out raster. Shape: [[[x_min, x_max], [y_min, y_max]], ... n-bboxes]
    :param topo_ras_buf_d: buffer distance in units of the crs to apply to the topographic raster
    :param topo_ras_dims: dimensions of the topographic raster in units of the crs
    :param topo_ras_fpath: absolute filepath to the topographic raster used for model input
    :param landsat_ras_fpath: absolute filepath to the landsat raster used for model input
    :param out_ras_dir: directory to save output raster
    :param pred_ras_res: resolution of output raster in units of the crs
    config file loaded from the checkpoint directory
    :param progress_bar: whether to display a progress bar during pytorch lightning prediction
    :param comp_names: list of component names to be used in the output raster
    :return: list of predictions for each bounding box

    """

    # Get the run name from checkpoint dir for naming out raster
    run_name = os.path.basename(checkpoint_dir)

    # Get tile name from lidar fpath
    tile_name = os.path.basename(las_fpath).replace('.las', '')

    # Set out out raster fpath
    out_ras_fpath = os.path.join(out_ras_dir, f"pred-tile_{tile_name}-run_nm_{run_name}.tif")

    # Set cols to extract from
    col_names = list(chain(*[[f"{nm}_pred"] for nm in comp_names]))

    # Set up data module with same params
    data_module = InferenceDataModule(
                        cfg=cfg,
                        bboxes=bboxes,
                        las_fpath=las_fpath,
                        device_id=device_id,
                        topo_ras_buf_d=topo_ras_buf_d,
                        topo_ras_dims=topo_ras_dims,
                        topo_ras_fpath=topo_ras_fpath,
                        landsat_fpath=landsat_ras_fpath)

    # Generate predictions for the lidar tile
    df = model_predict(lightning_model=lightning_model,
                       data_module=data_module,
                       ckpt_file=ckpt_file,
                       device_id=device_id,
                       return_id_or_bbox_center='bbox_center',
                       crs=crs,
                       progress_bar=progress_bar)

    # Rasterize the points to a geotiff
    rasterize_gridded_points(centroids=df, out_ras_fpath=out_ras_fpath,
                             col_names=col_names,
                             out_resolution=pred_ras_res, record_time=False)


def list_predict_to_raster(clean_las_fpaths, device_id, checkpoint_dir, bboxes, topo_ras_buf_d, topo_ras_dims, topo_ras_fpath, landsat_ras_fpath, crs,
                           out_ras_dir, pred_ras_res, batch_size):
    """
    Loops through a list of cleaned lidar tiles and generates a raster for each tile using a pytorch lightning model
    :param clean_las_fpaths: list of absolute filepaths to input lidar tiles
    :param device_id: Integer indicating GPU device to use (e.g., 0)
    :param checkpoint_dir: Directory where config and model checkpoint are stored for the target trained model
    :param bboxes: array of bboxes for each pixel in out raster. Shape: [[[x_min, x_max], [y_min, y_max]], ... n-bboxes]
    :param topo_ras_buf_d: buffer distance in units of the crs to apply to the topographic raster
    :param topo_ras_dims: dimensions of the topographic raster in units of the crs
    :param topo_ras_fpath: absolute filepath to the topographic raster used for model input
    :param landsat_ras_fpath: absolute filepath to the landsat raster used for model input
    :param crs: coordinate reference system of input lidar tile and generated raster
    :param out_ras_dir: directory to save output rasters
    :param pred_ras_res: resolution of output raster in units of the crs
    :param batch_size: number of samples to pass through model at once

    :return:
    """
    # If you do not have a trainer and model available, provide the directory where config and model checkpoint are stored
    cfg_files = list(Path(checkpoint_dir).glob("*" + 'yaml'))
    ckpt_files = list(Path(checkpoint_dir).glob("*" + 'ckpt'))

    assert len(cfg_files) == 1, print(f"Multiple config files present:\n{cfg_files}")
    assert len(ckpt_files) == 1, print(f"Multiple checkpoint files present:\n{ckpt_files}")

    # Select the config and model checkpoint files
    cfg_file = cfg_files[0]
    ckpt_file = ckpt_files[0]

    # Load config from checkpoint dir
    with open(cfg_file, "r") as yamlfile:
        cfg = yaml.load(yamlfile, Loader=yaml.FullLoader)

    # Overwrite batch size if specified
    if batch_size is not None:
        cfg['batch_size'] = batch_size

    # Rebuild model for inference using config, we do not need PlotIDs returned as we are not using a test dataset
    model = LitModel(cfg, return_id_or_bbox_center='bbox_center')

    # Loop through all the cleaned lidar tiles and generate one output raster for each tile
    for las_fpath in tqdm(clean_las_fpaths, desc=f"GPU - {device_id}: Predicting to rasters...", colour="cyan"):
        predict_to_raster(
            lightning_model=model,
            ckpt_file=ckpt_file,
            cfg=cfg,
            las_fpath=las_fpath,
            device_id=device_id,
            checkpoint_dir=checkpoint_dir,
            crs=crs,
            bboxes=bboxes,
            topo_ras_buf_d=topo_ras_buf_d,
            topo_ras_dims=topo_ras_dims,
            topo_ras_fpath=topo_ras_fpath,
            landsat_ras_fpath=landsat_ras_fpath,
            out_ras_dir=out_ras_dir,
            pred_ras_res=pred_ras_res,
            progress_bar=False)


if __name__ == "__main__":

    RUN_IN_PARALLEL = True

    # Load the ensemble training logs to get the model save directories
    train_logs = pd.read_csv(r"")

    #Classify model types
    train_logs['model_type'] = train_logs.apply(lambda row: check_model_type(row), axis=1)

    #Only use models train/tested on fold 1, and pick the best performing run on the test set per variant
    train_logs = train_logs[(train_logs['data_fold'] == 1)]

    #Sort the logs by tree rmse and take the best performing run per model type
    train_logs = train_logs.sort_values(by='tree_rmse', ascending=True).groupby('model_type').head(1)

    #Subset to ALS, ALS+Spectral+Thermal, and Full models only
    train_logs = train_logs[train_logs['model_type'].isin(['L', 'Full'])]

    #Save info for selected inference model training runs
    train_logs.to_csv(r"", index=False)

    # Get a list of the checkpoint directories
    checkpoint_dir_list = train_logs['save_dir'].tolist()

    # Set misc args
    batch_size = 512
    bbox_shp_out_fpath = None

    #Raster processing args
    pred_ras_res = 22.56
    crs = "EPSG:2953"
    base_ras_dir = r''

    # Lidar args
    clean_las_dir = r""
    lidar_index_fpath = r""

    # Topographic raster params
    topo_ras_fpath = r""
    topo_ras_dims = (14, 14)
    topo_ras_buf_d = 2

    #Multispectral args
    landsat_ras_fpath =  r""

    # Get the cleaned lidar file paths
    all_clean_las_fpaths = list(Path(clean_las_dir).glob("*.las"))

    # Get the lidar data bounding box coordinates (need to use all cleaned lidar files to build complete grid)
    lidar_bounds = get_las_bbox(all_clean_las_fpaths)

    bboxes = generate_bboxes(x_min=lidar_bounds[0],
                             x_max=lidar_bounds[2],
                             y_min=lidar_bounds[1],
                             y_max=lidar_bounds[3],
                             cell_size=pred_ras_res,
                             out_shp_fpath=bbox_shp_out_fpath
                             )

    # Record start time
    start_time = time.time()

    for idx, checkpoint_dir in tqdm(enumerate(checkpoint_dir_list), desc="Applying models..."):

        run_name = os.path.basename(checkpoint_dir)

        print(f"Applying model {run_name}, which is {idx + 1}: of {len(checkpoint_dir_list)} models.")

        # Create dir for run name if it does not yet exist
        out_ras_dir = os.path.join(base_ras_dir, run_name)
        if not os.path.exists(out_ras_dir):
            os.makedirs(out_ras_dir)

        # Read files that have already been processed
        out_ras_fname_ls = [os.path.basename(f).split(".")[0] for f in list(Path(out_ras_dir).glob("*.tif"))]

        # Extract part of filename with lidar tile name
        processed_tiles = [f.split("-")[1].replace("tile_", "") + ".las" for f in out_ras_fname_ls]

        target_clean_las_fpaths = [f for f in all_clean_las_fpaths if os.path.basename(f) not in processed_tiles]

        print(f"Already processed {len(processed_tiles)} tiles, {len(target_clean_las_fpaths)} tiles remaining")

        # Get the available GPU device names
        cuda_ids = [i for i in range(torch.cuda.device_count())]

        if RUN_IN_PARALLEL:

            # Split las fpaths into N GPU lists for parallel processing
            n_gpus = len(cuda_ids)
            fpaths_ls_ls = [target_clean_las_fpaths[i::n_gpus] for i in range(n_gpus)]

            # Combine all arguments to be used in starmap
            args = [(fpaths_ls,
                    device_id,
                    checkpoint_dir,
                    bboxes,
                    topo_ras_buf_d,
                    topo_ras_dims,
                    topo_ras_fpath,
                    landsat_ras_fpath,
                    crs,
                    out_ras_dir,
                    pred_ras_res,
                    batch_size) for
                    fpaths_ls, device_id in zip(fpaths_ls_ls, cuda_ids)]

            # Run the predict to raster function in parallel
            print("Starting parallel processing...")

            with mp.Pool(n_gpus) as p:
                p.starmap(list_predict_to_raster, args)

            print("Parallel processing complete!")

        else:

            print("Starting sequential processing...")

            list_predict_to_raster(
                clean_las_fpaths=target_clean_las_fpaths,
                device_id=0,
                checkpoint_dir=checkpoint_dir,
                bboxes=bboxes,
                topo_ras_buf_d=topo_ras_buf_d,
                topo_ras_dims=topo_ras_dims,
                topo_ras_fpath=topo_ras_fpath,
                landsat_ras_fpath=landsat_ras_fpath,
                crs=crs,
                out_ras_dir=out_ras_dir,
                pred_ras_res=pred_ras_res,
                batch_size = batch_size
            )

    # Report runtime
    end_time = time.time()
    runtime_seconds = end_time - start_time
    if RUN_IN_PARALLEL:
        print("\nFinished parallel processing.\n")
    print(f"\nTime required:\n{round((runtime_seconds / 60), 2)} minutes for {len(all_clean_las_fpaths)} unique tiles x {len(checkpoint_dir_list)} models,\nfor a total of {len(all_clean_las_fpaths) * len(checkpoint_dir_list)} tiles.")
