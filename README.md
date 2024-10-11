# Forest Biomass Estimation Using Deep Learning Data Fusion of LiDAR, Multispectral, and Topographic Data

<img src="images/fusion_model_architecture.png" alt= "fusion_model_architecture">

## Citation
...


## Key Features:
- **Data Augmentation**: Includes spectral and terrain data augmentation techniques.
- **Multi-Branch CNN Architecture**: Supports 1-D CNN for spectral data, 2-D CNN for topographic data, and Octree-CNN for lidar data.
- **Configurable Hyperparameters**:  Adjust model and training parameters via `config.yaml`.
- **Training and Evaluation**: Includes setup for training, validation, testing, and inference.

## Setup Instructions:
1. **Clone the Repository**:
   ```sh
   git clone https://github.com/harryseely/Forest-Biomass-Data-Fusion.git
   cd Forest-Biomass-Data-Fusion
   ```

2. **Install Dependencies**:
   ```sh
   pip install -r requirements.txt
   ```

3. **Configure the Project**:
   - Edit `config.yaml` to set your desired parameters and file paths.

4. **Run Training**:
   ```sh
   train_model.py
   ```

## License:
This project is licensed under the MIT License. See the `LICENSE` file for more details.

**Full Changelog**: https://github.com/harryseely/Forest-Biomass-Data-Fusion/commits/v1.0.0

## Referenced Repositories
The following GitHub repos were essential in the development of the code used in this study:

- [OCNN-Pytorch](https://github.com/octree-nn/ocnn-pytorch)
- [1-D CNN For Global Canopy Height Regression](https://github.com/langnico/GEDI-BDL)



 