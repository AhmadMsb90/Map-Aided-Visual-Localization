import os
from pathlib import Path
import cv2
import numpy as np
import torch

# --- Algorithm & Matcher Scaling Thresholds ---
MIN_GOOD_MATCHES = 8
RANSAC_REPROJ_THRESHOLD = 5.0
MAX_DIM = 1024
MAX_SP_SIZE = 640

# --- Reproducibility Seeds ---
cv2.setRNGSeed(0)
np.random.seed(0)

# --- Hardware Acceleration Device ---
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# --- Directory Paths & Model File Handles ---
BASE_PATH = Path("../Models/superpoint_superglue_project")
SP_ROOT = BASE_PATH / "superpoint_project2" / "pytorch-superpoint"
SP_MODELS = SP_ROOT / "models"
SP_UTILS = SP_ROOT / "utils"
SP_WEIGHTS = "../Models/superpoint_superglue_project/superpoint_v1.pth"

ORTHO_PATH = "../Maps/rgb_wide_images-only_orthomosaic.tif"
DSM_PATH = "../Maps/rgb_wide_images-only_dsm.tif"
IMAGE_DIR = "../Field_D_13_images/extracted_images2/"
META_PATH = "../Field_D_13_images/extracted_images2/metadata_with_euler.csv"
VIS_DIR = "../matches_out_sp_bf_both_align/"

# --- UAV Target Spatial Resolutions ---
UAV_IMAGE_RES_W = 1936
UAV_IMAGE_RES_H = 1460
GSD_UAV_X_M = 0.6 / 100.0
GSD_UAV_Y_M = 0.6 / 100.0

# --- Intrinsic Matrices & Frame Orientation Rotations ---
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