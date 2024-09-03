# Other packages
import os
from pathlib import Path
import pytorch_lightning as pl
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
import warnings
import rasterio
import torchvision.transforms.functional as ttf
import wandb

# My base
from utils.ocnn_custom_utils import CustomCollateBatch
from utils.data_utils import load_octree_sample, augment_spectral_sample, read_las_to_np


# Class used to create an octree dataset
class BiomassDataset(Dataset):
    """Point cloud dataset where one sample is a file."""

    def __init__(self, cfg, split="train", data_fold=1, logger=None):

        """
        IMPORTANT: format of np array storing lidar data returned has the following format:
        N x C
        Where N is the numer of points and C is the number of columns
        The columns included in the np array by index are:
            0 - X
            1 - Y
            2 - Z

           *If additional features are used ('ocnn_use_feats' in cfg )
            3 - Intensity (Normalized)
            4 - Return Number
            5 - Classification
            6 - Scan Angle Rank
            7 - Number of Returns

            *If normals are estimated ('use_normals' in cfg )
            8 - x component of normals
            9 - y component of normals
            10 - z component of normals

        Spectral data used in CNN1D is stored in a 1D np array with a length of N (where N is the number of bands)

        """

        if split == "train" and cfg['augment']:
            self.augment = True
        else:
            self.augment = False

        # Attach fold number
        self.data_fold = data_fold

        # Filter which files to use ------------------------------------------------------------------------------------

        # Get a list of all files in the directory
        pc_data_path = os.path.join(cfg['data_dir'], cfg['dataset'], "lidar")
        self.las_files = list(Path(pc_data_path).glob("*" + 'las'))
        assert len(self.las_files) > 0

        # Load biomass data reference data -----------------------------------------------------------------------------
        ref_data_path = os.path.join(cfg['data_dir'], cfg['dataset'], 'biomass_labels.csv')
        self.df = pd.read_csv(ref_data_path, sep=",", header=0)

        # If only training with a subset of data (data_partition), randomyl select a subset of the data
        if cfg['data_partition'] < 1:
            self.df = self.df.sample(frac=cfg['data_partition'])

        # Ensure plot IDs are strings
        self.df['PlotID'] = self.df['PlotID'].astype(str)

        # Attach cfg ---------------------------------------------------------------------------------------------------
        self.cfg = cfg

        # Attach the data split to use
        self.split = split

        # Create a df with las files and join with plot DF
        las_files_df = pd.DataFrame(self.las_files, columns=["las_fpath"])
        las_files_df['PlotID'] = las_files_df['las_fpath'].apply(lambda x: str(os.path.basename(x).split(".")[0]))
        self.df = pd.merge(self.df, las_files_df, on='PlotID', how='inner')

        # Use current fold column to filter df to target split
        fold_col = f"fold_{self.data_fold}"
        self.df = self.df[self.df[fold_col] == self.split]

        # Get target las files for split and corresponding plot ids
        self.las_files = self.df['las_fpath'].tolist()
        self.plot_ids = self.df['PlotID'].tolist()

        # Ensure df is in same order as files
        self.df.index = self.df['PlotID']
        self.df = self.df.reindex(self.plot_ids)
        self.df.reset_index(drop=True, inplace=True)

        # Select spectral data to be used in model ----------------------------------------------------------------------

        # Select spectral band columns to be used in model
        self.spectral_cols = cfg['bands'].split(",")

        # Whether to include thermal
        if cfg['include_thermal'] is False:
            self.spectral_cols = [c for c in self.spectral_cols if 'thermal' not in c]

        # Update selected bands in logger
        if logger is not None:
            selected_bands = ",".join(self.spectral_cols)
            wandb.config.update({"bands": selected_bands}, allow_val_change=True)

        print(f"Selected bands\n:{self.spectral_cols}")

        # Get standard deviation (sd) for all bands for training set samples -------------------------------------------
        self.spec_sd = self.df.loc[:, self.spectral_cols].std().to_dict()

        # Load las files into memory -----------------------------------------------------------------------------------
        if self.cfg['ocnn_lenet']:

            self.point_clouds = {}

            for f in tqdm(self.las_files, leave=False, position=0, disable=not cfg['verbose'],
                          desc=f"Loading las files into memory for {split} set..."):
                # Read las file to np array
                pc = read_las_to_np(str(f),
                                    use_ground_points=cfg['use_ground_points'],
                                    centralize_coords=True,
                                    compute_normals=cfg['use_normals'])

                # Add to point cloud dict
                plot_id = os.path.basename(f).split(".")[0]
                self.point_clouds[plot_id] = pc

        # Load terrain rasters into memory ---------------------------------------------------------------------------------
        if cfg['terrain_cnn']:

            dem_data_dir = os.path.join(cfg['data_dir'], cfg['dataset'], 'terrain_rasters')

            # Set terrain raster filepaths using plot ids list
            terrain_ras_fnames = [f"terrain_{plot_id}.tif" for plot_id in self.plot_ids]
            self.terrain_files = [os.path.join(dem_data_dir, f) for f in terrain_ras_fnames]

            assert len(self.terrain_files) > 0

            self.terrain_rasters = {}

            plot_id_ls = []
            terrain_raster_ls = []
            for f in tqdm(self.terrain_files, leave=False, position=0, disable=not cfg['verbose'],
                          desc=f"Loading DEM rasters into memory for {split} set..."):
                with rasterio.open(f, 'r') as src:
                    terrain_ras = src.read()  # Use masked=True to mask nodata values

                    # terrain raster needs to be in the format H x W x C
                    # Move axes of input terrain to match this shape
                    terrain_ras = np.moveaxis(terrain_ras, 0, -1)

                # Check for na vals
                if np.isnan(terrain_ras).any():
                    raise ValueError(f"Terrain raster contains NA values: {f}")

                plot_id_ls.append(os.path.basename(f).split(".")[0])

                terrain_raster_ls.append(terrain_ras)

            print("test")

            #Get the sd for each band across all input rasters
            ras_stacked = np.stack(terrain_raster_ls)

            ras_vals = np.reshape(ras_stacked,
                       (ras_stacked.shape[0] * ras_stacked.shape[1] * ras_stacked.shape[2], ras_stacked.shape[3]))

            self.terrain_sd = np.std(ras_vals, axis=0)

            #Convert ras list into a dict with corresponding plot id
            self.terrain_rasters = dict(zip(plot_id_ls, terrain_raster_ls))

        super().__init__()

    def __len__(self):
        return len(self.las_files)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        # Load biomass data --------------------------------------------------------------------------------------------

        # Get plot ID from filename
        PlotID = self.plot_ids[idx]

        # Extract matching row from self.df
        sample_row = self.df.loc[self.df["PlotID"] == PlotID]

        # Extract bark, branch, foliage, wood values for the correct sample
        if self.cfg['z_score']:
            bark = sample_row["bark_z"].values[0]
            branch = sample_row["branch_z"].values[0]
            foliage = sample_row["foliage_z"].values[0]
            wood = sample_row["wood_z"].values[0]

        else:
            bark = sample_row["bark_Mg_ha"].values[0]
            branch = sample_row["branch_Mg_ha"].values[0]
            foliage = sample_row["foliage_Mg_ha"].values[0]
            wood = sample_row["wood_Mg_ha"].values[0]

        # Combine z targets into a list
        target = np.array([bark, branch, foliage, wood])

        sample = dict()

        # Add label and PlotID
        sample['target'] = torch.from_numpy(target).float()
        sample['PlotID'] = PlotID

        if self.cfg['ocnn_lenet']:

            # Select point clouds
            pc = self.point_clouds[PlotID]

            # Set sample dict key for whether the input contains points
            sample['contains_points'] = True

            # Load octree sample from point cloud --------------------------------------------------------------------------
            if pc.shape[0] == 0:
                # Add a warning that no points were found
                warnings.warn(f"PlotID: {PlotID} contains zero points. Adding a dummy point cloud.")

                # Flag sample as not containing points
                sample['contains_points'] = False

                # If pc is empty, create a dummy pc as a placeholder with 1 point
                pc = np.ones([1, 11])

            # Load octree sample from point cloud --------------------------------------------------------------------------
            octree_dict = load_octree_sample(pc, cfg=self.cfg, idx=idx, augment=self.augment)

            # Add octree info to sample
            sample.update(octree_dict)

        # Load spectral data sample ------------------------------------------------------------------------------------
        if self.cfg['spec_cnn']:

            # Select spectral data
            spectral_vals = sample_row[self.spectral_cols]

            # Apply augmentation to spectral data if specified
            if self.augment:
                spectral_vals = augment_spectral_sample(spectra_row=spectral_vals, spec_sd=self.spec_sd,
                                                        sd_scaling_factor=0.1)

            # Convert spectral values to np array
            spectral_vals = spectral_vals.to_numpy()

            # Set input sample to contain landsat spectral values
            sample['spectral'] = torch.from_numpy(spectral_vals).float()

        # Load terrain raster data sample -----------------------------------------------------------------------------------------
        if self.cfg['terrain_cnn']:

            # Select terrain raster
            terrain_ras = self.terrain_rasters[f"terrain_{PlotID}"]

            #Convert to tensor
            terrain_tensor = ttf.to_tensor(terrain_ras)

            #Apply gaussian blur to each band separately
            if self.augment:

                # Apply gaussian blur to each band separately
                if self.augment:
                    for band in range(terrain_tensor.shape[0]):

                        sd_scaled = self.terrain_sd[band] * 0.1

                        band_noise = np.random.normal(loc=0.0, scale=sd_scaled,
                                                      size=(terrain_tensor.shape[1], terrain_tensor.shape[2]))

                        terrain_tensor[band] =  terrain_tensor[band] + band_noise

            sample['terrain'] = terrain_tensor

        return sample


class BiomassDataModule(pl.LightningDataModule):
    def __init__(self, cfg, logger=None):
        super().__init__()
        self.cfg = cfg
        self.logger = logger
        if self.cfg['ocnn_lenet']:
            self.collate_fn = CustomCollateBatch(batch_size=cfg['batch_size'])
        else:
            self.collate_fn = None

    def prepare_data(self):
        # Currently this method is not being used
        pass
        # Possible uses:
        # Download and save dataset here
        # This can also be used to do any process that should not be parallelized and only run once

    def setup(self, stage: str):
        if stage == "fit":
            self.train_dataset = BiomassDataset(self.cfg, split="train", data_fold=self.cfg['data_fold'],
                                                logger=self.logger)
            self.val_dataset = BiomassDataset(self.cfg, split="val", data_fold=self.cfg['data_fold'],
                                              logger=self.logger)

        if stage == "test":
            self.test_dataset = BiomassDataset(self.cfg, split="test", data_fold=self.cfg['data_fold'],
                                               logger=self.logger)

        # Currently predict stage is only set up to use test dataset
        if stage == "predict":
            self.predict_dataset = BiomassDataset(self.cfg, split="test", data_fold=self.cfg['data_fold'])

    def train_dataloader(self):
        return DataLoader(self.train_dataset, batch_size=self.cfg['batch_size'], shuffle=True, pin_memory=True,
                          collate_fn=self.collate_fn, drop_last=True, num_workers=self.cfg['n_workers'])

    def val_dataloader(self):
        return DataLoader(self.val_dataset, batch_size=self.cfg['batch_size'],
                          pin_memory=True, collate_fn=self.collate_fn, drop_last=True,
                          num_workers=self.cfg['n_workers'])

    def test_dataloader(self):
        return DataLoader(self.test_dataset, batch_size=self.cfg['batch_size'],
                          pin_memory=True, collate_fn=self.collate_fn, num_workers=self.cfg['n_workers'])

    def predict_dataloader(self):
        return DataLoader(self.predict_dataset, batch_size=self.cfg['batch_size'],
                          pin_memory=True, collate_fn=self.collate_fn, num_workers=self.cfg['n_workers'])