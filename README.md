# UAV Pose Estimation

## Overview
This project investigates a map-aided UAV localization framework based on 3D–2D registration for absolute pose estimation. UAV images are matched against orthomosaic map patches to establish correspondences, while elevation information is obtained from a Digital Surface Model (DSM) to form 3D points. The system estimates UAV camera pose by matching aerial images with a georeferenced map and solving 2D–3D correspondences.

## Technical Pipeline & Mathematical Formulation
The registration and estimation workflow proceeds through a structured geometric sequence designed to interface directly with projected map coordinate systems:

1. **Spatial Cropping & GSD Alignment:** The initial UAV geodetic telemetry (WGS84) is transformed into the ETRS89 / UTM 32N projected coordinate system (EPSG:25832). A map Region of Interest (ROI) is cropped from the global orthomosaic based on the Ground Sampling Distance (GSD) ratios, ensuring physical features share a uniform scale across both fields of view.

2. **2D–3D World Correspondence Generation:** Following feature matching and outlier rejection via a RANSAC-driven homography matrix, the 2D pixel coordinates $(u, v)$ of the inlier keypoints within the orthomosaic patch are georeferenced into absolute map metrics. Elevation values $(Z)$ are sampled directly at those spatial coordinates from the co-registered Digital Surface Model (DSM) raster data to build an absolute 3D point cloud. For mathematical stability during numerical solver execution, these world points are shifted into a local frame centered at the initial RTK-GPS coordinate prior.

3. **PnP Solver Initializations:** Depending on the configured strategy, the Perspective-n-Point (PnP) solver is seeded with one of two distinct canonical frames to initialize iterative optimization:

    * **Anchor-based Initialization:** A deterministic heuristic rotation matrix $\mathbf{R}_{\text{anchor}}$ is utilized to establish a stable down-looking nadir baseline configuration by swapping the horizontal coordinate components and flipping the vertical reference vector:
    $$\mathbf{R}_{\text{map} \rightarrow \text{cam}}^{\text{init}} = \mathbf{R}_{\text{anchor}}$$

    * **Prior-based Initialization (IMU):** The orientation matrix $\mathbf{R}_{\text{imu}}$ derived from onboard telemetry is treated as directly absolute relative to the global map grid coordinate system ($\mathbf{R}_{\text{map} \rightarrow \text{body}}$). The seed prior is formulated by chaining this attitude state with the constant structural camera calibration displacement profile ($\mathbf{R}_{\text{boresight}}$):
    $$\mathbf{R}_{\text{map} \rightarrow \text{cam}}^{\text{prior}} = \mathbf{R}_{\text{boresight}} \cdot \mathbf{R}_{\text{imu}}$$

4. **Pose Optimization & Camera Position Recovery:** The solver refines the initialized pose guess against the local 3D-2D points, producing an optimized camera transformation matrix $\mathbf{R}_{\text{map} \rightarrow \text{cam}}^{\text{refined}}$ and translation vector $\mathbf{t}$. The spatial global camera location vector $\mathbf{C}$ is subsequently recovered in absolute metrics using spatial frame inversion:

$$\mathbf{C} = -\left(\mathbf{R}_{\text{map} \rightarrow \text{cam}}^{\text{refined}}\right)^\top \mathbf{t} + \mathbf{C}_{\text{init}}$$

### Evaluated Experimental Configurations
The localization pipeline follows a common structure across all configurations, consisting of feature extraction, feature matching, image alignment, and pose estimation using a PnP solver. The evaluated setups differ in the choice of feature detector, matching strategy, and the use of rotation alignment derived from IMU data.

Classical and learning-based methods are compared under the same conditions. For SIFT, no explicit image rotation is applied. In contrast, learned feature methods (SuperPoint and DISK) use a yaw-based rotation derived from IMU measurements to approximately align UAV images with the orthomosaic, improving correspondence consistency under large viewpoint changes.

For matching, both classical (brute-force with KNN and Lowe’s ratio test) and learning-based approaches (SuperGlue and LightGlue) are evaluated. Finally, camera pose is estimated using a PnP formulation with two initialization strategies: a heuristic anchor-based rotation and an IMU-based prior.

The following table summarizes all evaluated configurations:

| Method | Rotation | Matcher | PnP Init |
| :--- | :--- | :--- | :--- |
| SIFT | None | BF + KNN (Ratio) | Anchor / IMU |
| SuperPoint (default) | Yaw | SuperGlue (default) | Anchor / IMU |
| SuperPoint (trained) | Yaw | BF + KNN (Ratio) | Anchor / IMU |
| DISK (default) | Yaw | LightGlue (default) | Anchor / IMU |

---

## Project Structure
The repository is modularized into dedicated execution components for each localization framework alongside spatial maps and utility toolkits:

```bash
.
├── calibration
│   ├── calibrate.ipynb
│   └── calibration images
├── Field_D_13_images
│   ├── extracted_images2
│   │   └── metadata_with_euler.csv
│   │   └── (place Field D images here)
│   ├── plot_trajectory.ipynb
│   └── trajectory.png
├── Field_E_images_metadata
│   └── images
│       └── metadata_with_euler.csv
│       └── (place Field E images here)
├── Maps
│   └── (place all DSM / orthomosaic / map files here)
├── Models
│   └── superpoint_superglue_project
│       ├── SuperGluePretrainedNetwork
│       ├── LightGlue
│       └── superpoint_project2
├── pose_estimation_disk_lightglue_align
│   ├── config.py
│   ├── pipeline.py
│   ├── utils.py
│   └── main.py
├── pose_estimation_sift
│   ├── config.py
│   ├── pipeline.py
│   ├── utils.py
│   └── main.py
├── pose_estimation_sup_glue_align
│   ├── config.py
│   ├── pipeline.py
│   ├── utils.py
│   └── main.py
├── README.md
├── README.pdf
└── Results
    ├── AM_RnD_experiment.csv
    ├── ground_truth.csv
    ├── ground_truth_with_gps.csv
    ├── uav_images_with_clusters.csv
    ├── uav_pose_disk_lihtglue.csv
    ├── uav_pose_results_pt_trained_BF.csv
    ├── uav_pose_results_sift_align.csv
    ├── uav_pose_results_sift.csv
    └── uav_pose_results_sup_glue.csv
```

## Dependencies

### Core Libraries
* numpy
* pandas
* opencv-python
* opencv-contrib-python
* matplotlib
* scipy
* scikit-learn
* scikit-image

### Deep Learning
* torch
* torchvision
* tensorflow

### Geospatial & Math
* rasterio
* pyproj
* affine
* shapely
* pymap3d

### Image Processing & Utils
* tqdm
* Pillow
* imageio
* imgaug

### Interactive Environment
* jupyter
* ipykernel
* notebook

### Additional
* protobuf
* h5py
* networkx
* joblib
* requests
* pyyaml
* sympy

---

## SuperPoint Training
For SuperPoint training and setup, follow the official repository:
https://github.com/magicleap/SuperPointPretrainedNetwork

This includes:
- Training pipeline
- Pretrained model usage
- Dataset preparation
- Evaluation scripts

---

## Models and Implementations
The following repositories were used for feature extraction, matching, and training:
- SuperPoint (training implementation):  
  https://github.com/eric-yyjau/pytorch-superpoint  
- SuperPoint + SuperGlue (pretrained models and pipeline):  
  https://github.com/magicleap/SuperGluePretrainedNetwork  
- DISK and LightGlue (feature extraction and matching):  
  https://github.com/cvg/lightglue  

---



## Data & Checkpoints Availability
The repository contains only code, evaluation outputs, and processed results. Large-scale training datasets, intermediate checkpoints, and full experimental logs were removed to reduce storage requirements.

This includes:
- Custom SuperPoint training dataset:
  - `Models/superpoint_superglue_project/superpoint_project2/pytorch-superpoint/datasets/my_custom_data/`
  - Removed contents: images (~6.2 GB), images_clean (~733 MB), labels (~13 MB)
- Training logs and checkpoints:
  - `Models/superpoint_superglue_project/superpoint_project2/pytorch-superpoint/logs/`
  - Removed contents: model checkpoints, prediction outputs, and runs.
- Additional runtime artifacts:
  - `runs/`, `pretrained/`, and intermediate prediction dumps.

### Data Access (Google Drive)
All large datasets, maps, and training artifacts that are not included in this repository can be accessed via the following Google Drive link:

[Data on the Google Drive](https://drive.google.com/drive/folders/1qMgkUY4-T-UQzsjNcOpKW4GWilLFfzEj?usp=sharing)

The folder contains the following compressed archive files:
- `calibration.zip`: UAV camera calibration images and notebook files.
- `logs.zip`: Complete SuperPoint training logs, benchmarks, and run histories.
- `Maps.zip`: Full Digital Surface Model (DSM) and orthomosaic raster map layers.
- `Models.zip`: Deep learning network structure models, repositories, and weights.
- `my_custom_data.zip`: Custom SuperPoint training imagery and matching dataset records.
