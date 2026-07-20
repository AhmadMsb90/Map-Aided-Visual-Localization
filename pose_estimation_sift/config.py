import numpy as np
import cv2

# --- Algorithmic Feature & Matching Parameters ---
MAX_FEATURES = 8000            # Max SIFT keypoints to detect
RATIO_TEST = 0.75              # Lowe's ratio test threshold
MIN_GOOD_MATCHES = 8           # Minimum matches needed to proceed
RANSAC_REPROJ_THRESHOLD = 5.0  # Homography RANSAC outlier threshold (pixels)
MAX_DIM = 1600                 # Maximum dimension for scaling images

# --- Seeds ---
cv2.setRNGSeed(0)
np.random.seed(0)

# --- Path Configurations ---
ORTHO_PATH = "../Maps/rgb_wide_images-only_orthomosaic.tif"
DSM_PATH = "../Maps/rgb_wide_images-only_dsm.tif"
IMAGE_DIR = "../Field_D_13_images/extracted_images2/"
META_PATH = "../Field_D_13_images/extracted_images2/metadata_with_euler.csv"

# --- Drone Spatial Parameters ---
UAV_IMAGE_RES_W = 1936
UAV_IMAGE_RES_H = 1460
GSD_UAV_X_M = 0.6 / 100.0
GSD_UAV_Y_M = 0.6 / 100.0

# --- Camera Intrinsics & Calibration Matrices ---
K = np.array([
    [1806.84457,    0.0,     945.416679],
    [   0.0,     1806.43529, 738.031572],
    [   0.0,        0.0,        1.0    ]
], dtype=np.float64)

DIST_COEFFS = np.array([
    -0.20484306, 0.1278626, -0.00041228, 0.0005107, -0.06076351
]).reshape(-1, 1)

# --- Geometric Correction Transformations ---
# Reference initial map guess alignment and camera body correction alignment matrices
R_ANCHOR = np.array([[0, 1, 0], [1, 0, 0], [0, 0, -1]], dtype=np.float64)
R_BORESIGHT = np.array([[0, 1, 0], [-1, 0, 0], [0, 0, -1]], dtype=np.float64)