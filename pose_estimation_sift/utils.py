import numpy as np
import cv2
import config

def to_gray(bgr):
    """Convert BGR image to grayscale."""
    return cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)

def resize_keep(img, max_dim=config.MAX_DIM):
    """Resize image down proportionally if its maximum side exceeds max_dim."""
    h, w = img.shape[:2]
    scale = 1.0
    if max(h, w) > max_dim:
        scale = max_dim / float(max(h, w))
        img = cv2.resize(img, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_AREA)
    return img, scale

def get_euler(R):
    """Extract Euler angles (Yaw, Pitch, Roll in degrees) from a 3x3 rotation matrix."""
    sy = np.sqrt(R[0, 0]**2 + R[1, 0]**2)
    if sy > 1e-6:
        return np.degrees([
            np.arctan2(R[1, 0], R[0, 0]),   # Yaw
            np.arctan2(-R[2, 0], sy),       # Pitch
            np.arctan2(R[2, 1], R[2, 2])    # Roll
        ])
    return np.degrees([
        0, 
        np.arctan2(-R[2, 0], sy), 
        np.arctan2(-R[1, 2], R[1, 1])
    ])

def R_from_quaternion(w, x, y, z):
    """Convert Quaternion parameters to an orthonormal 3x3 orientation matrix."""
    n = np.sqrt(w*w + x*x + y*y + z*z)
    w, x, y, z = w/n, x/n, y/n, z/n
    return np.array([
        [1 - 2*(y*y + z*z),     2*(x*y - z*w),     2*(x*z + y*w)],
        [    2*(x*y + z*w), 1 - 2*(x*x + z*z),     2*(y*z - x*w)],
        [    2*(x*z - y*w),     2*(y*z + x*w), 1 - 2*(x*x + y*y)]
    ], dtype=np.float64)

def compute_reprojection_error(obj_pts, img_pts, rvec, tvec, K, dist):
    """Calculate mean and standard deviation of reprojection residues."""
    proj_pts, _ = cv2.projectPoints(obj_pts, rvec, tvec, K, dist)
    proj_pts = proj_pts.reshape(-1, 2)
    img_pts = img_pts.reshape(-1, 2)
    errors = np.linalg.norm(proj_pts - img_pts, axis=1)
    return errors.mean(), errors.std()