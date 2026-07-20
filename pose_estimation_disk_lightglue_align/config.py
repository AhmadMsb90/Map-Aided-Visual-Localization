import os
from pathlib import Path
import cv2
import numpy as np
import torch

# --- Algorithm & Feature Matching Parameters ---
MIN_GOOD_MATCHES = 8           # Minimum matches required to accept geometry 
RANSAC_REPROJ_THRESHOLD = 5.0  # Homography RANSAC filtering threshold in px 
MAX_DIM = 1024                 # Max image resolution dimension for downscaling 
DETECTION_THRESHOLD = 0.6      # DISK feature detector threshold 
MAX_NUM_KEYPOINTS = 4000       # Cap on keypoints extracted per crop 
PRUNING_KEYPOINT_THRESH = 0.2  # LightGlue early pruning threshold [cite: 2]

# --- Global Random Seeds ---
cv2.setRNGSeed(0)              # 
np.random.seed(0)              # 

# --- Runtime Device Configuration ---
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")  # 

# --- External Library and Path Configurations ---
SG_ROOT = Path("../Models/LightGlue")  # LightGlue local installation path 
ORTHO_PATH = "../Maps/rgb_wide_images-only_orthomosaic.tif"  
DSM_PATH = "../Maps/rgb_wide_images-only_dsm.tif"            
IMAGE_DIR = "../Field_D_13_images/extracted_images2/"        
META_PATH = "../Field_D_13_images/extracted_images2/metadata_with_euler.csv"  
VIS_DIR = "../matches_out_dsk_lg_both_default/"             

# --- Sensor Spatial Profiles ---
UAV_IMAGE_RES_W = 1936        
UAV_IMAGE_RES_H = 1460         
GSD_UAV_X_M = 0.6 / 100.0      
GSD_UAV_Y_M = 0.6 / 100.0      

# --- Camera Intrinsics & Extrinsics Alignment Matrices ---
K = np.array([
    [1806.84457,    0.0,     945.416679],
    [   0.0,     1806.43529, 738.031572],
    [   0.0,        0.0,        1.0    ]
], dtype=np.float64)  # [cite: 6]

DIST_COEFFS = np.array([
    -0.20484306, 0.1278626, -0.00041228, 0.0005107, -0.06076351
]).reshape(-1, 1)  

# Fixed initial PnP guesses and camera chassis boresight alignments
R_ANCHOR = np.array([[0, 1, 0], [1, 0, 0], [0, 0, -1]], dtype=np.float64)    
R_BORESIGHT = np.array([[0, 1, 0], [-1, 0, 0], [0, 0, -1]], dtype=np.float64) 