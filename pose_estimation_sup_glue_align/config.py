import os
from pathlib import Path
import cv2
import numpy as np
import torch

# --- Algorithm & Matching Thresholds ---
MIN_GOOD_MATCHES = 8
RANSAC_REPROJ_THRESHOLD = 5.0
MAX_DIM = 1024

# --- Reproducibility Seeds ---
cv2.setRNGSeed(0)
np.random.seed(0)

# --- Execution Device Configuration ---
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# --- Deep Learning Model Parameters ---
SG_ROOT = Path("../Models/superpoint_superglue_project/SuperGluePretrainedNetwork")
SUPERGLUE_CONFIG = {
    "superpoint": {
        "nms_radius": 4,
        "keypoint_threshold": 0.001,
        "max_keypoints": 4096
    },
    "superglue": {
        "weights": "outdoor",
        "sinkhorn_iterations": 20,
        "match_threshold": 0.1,
    },
}

# --- Paths and Directories ---
ORTHO_PATH = "../Maps/rgb_wide_images-only_orthomosaic.tif"
DSM_PATH = "../Maps/rgb_wide_images-only_dsm.tif"
IMAGE_DIR = "../Field_D_13_images/extracted_images2/"
META_PATH = "../Field_D_13_images/extracted_images2/metadata_with_euler.csv"
VIS_DIR = "../matches_out_sp_sg_both_align/"

# --- UAV Camera Operational Metadata ---
UAV_IMAGE_RES_W = 1936
UAV_IMAGE_RES_H = 1460
GSD_UAV_X_M = 0.6 / 100.0
GSD_UAV_Y_M = 0.6 / 100.0

# --- Intrinsic Matrices & Geometric Alignment Rotations ---
K = np.array([
    [1806.84457,    0.0,     945.416679],
    [   0.0,     1806.43529, 738.031572],
    [   0.0,        0.0,        1.0    ]
], dtype=np.float64)

DIST_COEFFS = np.array([
    -0.20484306, 0.1278626, -0.00041228, 0.0005107, -0.06076351
]).reshape(-1, 1)

R_ANCHOR = np.array([[0, 1, 0], [1, 0, 0], [0, 0, -1]], dtype=np.float64)
R_BORESIGHT = np.array([[0, 1, 0], [-1, 0, 0], [0, 0, -1]], dtype=np.float64)