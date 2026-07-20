import os
import sys
import types
import cv2
import numpy as np
import rasterio
from rasterio.transform import rowcol
from pyproj import Transformer
import torch
import config
import utils

# Initialize dynamic virtual module mock structures prior to mapping imports
models_pkg = types.ModuleType("models")
models_pkg.__path__ = [str(config.SP_MODELS)]
sys.modules["models"] = models_pkg

utils_pkg = types.ModuleType("utils")
utils_pkg.__path__ = [str(config.SP_UTILS)]
sys.modules["utils"] = utils_pkg

from models.SuperPointNet import SuperPointNet
from utils.utils import flattenDetection

class SuperPointBFPipeline:
    def __init__(self):
        if not os.path.exists(config.ORTHO_PATH):
            raise FileNotFoundError(f"Orthomosaic file not found: {config.ORTHO_PATH}")
            
        with rasterio.open(config.ORTHO_PATH) as ds:
            self.image_map_raw = ds.read(out_dtype="uint8")
            self.px2utm_transform = ds.transform
            self.ortho_crs = ds.crs
            self.bounds = ds.bounds
            
        self.GSD_x = abs(self.px2utm_transform.a)
        self.GSD_y = abs(self.px2utm_transform.e)
        
        if self.image_map_raw.shape[0] > 3:
            self.image_map_raw = self.image_map_raw[:3, :, :]
            
        image_map_rgb = np.transpose(self.image_map_raw, (1, 2, 0))
        self.image_map_bgr = cv2.cvtColor(image_map_rgb, cv2.COLOR_RGB2BGR)
        self.h_map, self.w_map = self.image_map_bgr.shape[:2]
        
        self.transformer = Transformer.from_crs("EPSG:4326", self.ortho_crs, always_xy=True)
        
        # Load and configure pretrained SuperPoint architecture
        self.net = SuperPointNet()
        self.net.load_state_dict(torch.load(config.SP_WEIGHTS, map_location=config.DEVICE))
        self.net = self.net.to(config.DEVICE).eval()
        
        # Setup cross-feature descriptor matching criteria
        self.bf = cv2.BFMatcher(cv2.NORM_L2, crossCheck=True)

    def extract_ortho_patch(self, lon, lat):
        """Extracts a matching geo-spatial window based on coordinates."""
        e_map, n_map = self.transformer.transform(lon, lat)
        row_f, col_f = rowcol(self.px2utm_transform, e_map, n_map)
        r, c = int(round(row_f)), int(round(col_f))
        
        ortho_per_uav_x = config.GSD_UAV_X_M / self.GSD_x
        ortho_per_uav_y = config.GSD_UAV_Y_M / self.GSD_y
        
        crop_w = int(config.UAV_IMAGE_RES_W * ortho_per_uav_x) // 2
        crop_h = int(config.UAV_IMAGE_RES_H * ortho_per_uav_y) // 2
        
        r1, r2 = max(r - crop_h, 0), min(r + crop_h, self.h_map)
        c1, c2 = max(c - crop_w, 0), min(c + crop_w, self.w_map)
        
        return self.image_map_bgr[r1:r2, c1:c2].copy(), r1, c1, e_map, n_map

    def run_superpoint(self, gray_img):
        """Extracts features and keypoints using SuperPoint."""
        h, w = gray_img.shape[:2]
        inp = gray_img.astype(np.float32) / 255.0
        inp = torch.from_numpy(inp).unsqueeze(0).unsqueeze(0).to(config.DEVICE)
        
        with torch.no_grad():
            semi, desc = self.net(inp)
            
        semi = semi.squeeze(0).cpu().numpy()
        desc = desc.squeeze(0).cpu().numpy()
        
        nodust = flattenDetection(semi, h, w)
        semi = nodust
        
        thresh = 0.015
        xs, ys = np.where(semi >= thresh)
        if len(xs) == 0:
            return np.zeros((0, 2), dtype=np.float32), np.zeros((0, 256), dtype=np.float32)
            
        scores = semi[xs, ys]
        
        # Sample deep network tracking descriptors down to target pixels
        kp = np.stack((ys, xs), axis=1).astype(np.float32)
        sampled_desc = []
        for i in range(len(kp)):
            cx, cy = int(kp[i][0]), int(kp[i][1])
            cx_d = min(max(int(round(cx / 8.0)), 0), desc.shape[2] - 1)
            cy_d = min(max(int(round(cy / 8.0)), 0), desc.shape[1] - 1)
            d_vec = desc[:, cy_d, cx_d]
            d_vec /= (np.linalg.norm(d_vec) + 1e-7)
            sampled_desc.append(d_vec)
            
        return kp, np.array(sampled_desc, dtype=np.float32)

    def save_matches_visualization(self, img1_rs, patch_bgr, img2_rs, mkpts0, mkpts1, uav_img_num):
        """Saves a verification canvas layout image to storage."""
        img_uav_vis = img1_rs.copy()
        patch_vis = img2_rs.copy()
        
        if len(img_uav_vis.shape) == 2:
            img_uav_vis = cv2.cvtColor(img_uav_vis, cv2.COLOR_GRAY2BGR)
        if len(patch_vis.shape) == 2:
            patch_vis = cv2.cvtColor(patch_vis, cv2.COLOR_GRAY2BGR)
            
        h1, w1 = img_uav_vis.shape[:2]
        h2, w2 = patch_vis.shape[:2]
        canvas = np.zeros((max(h1, h2), w1 + w2, 3), dtype=np.uint8)
        canvas[:h1, :w1] = img_uav_vis
        canvas[:h2, w1:w1+w2] = patch_vis
        
        for pt0, pt1 in zip(mkpts0, mkpts1):
            x0, y0 = int(pt0[0]), int(pt0[1])
            x1, y1 = int(pt1[0] + w1), int(pt1[1])
            color = tuple(np.random.randint(0, 255, 3).tolist())
            cv2.circle(canvas, (x0, y0), 3, color, -1)
            cv2.circle(canvas, (x1, y1), 3, color, -1)
            cv2.line(canvas, (x0, y0), (x1, y1), color, 1)
            
        os.makedirs(config.VIS_DIR, exist_ok=True)
        save_path = os.path.join(config.VIS_DIR, f"match_{uav_img_num}.png")
        cv2.imwrite(save_path, canvas)
        print("Saved debug verification matching canvas:", save_path)

    def process_frame(self, image_path, row_meta, uav_img_num):
        """Processes images, runs inference, tracks geometric features, and estimates poses."""
        uav_alt_gps = float(row_meta["gps_altitude"])
        uav_yaw = float(row_meta["yaw"])
        qx, qy, qz, qw = float(row_meta["imu_orientation_x"]), float(row_meta["imu_orientation_y"]), float(row_meta["imu_orientation_z"]), float(row_meta["imu_orientation_w"])
        
        img_uav_bgr = cv2.imread(image_path)
        if img_uav_bgr is None:
            return None
        img_uav_bgr = img_uav_bgr.astype("uint8")
        
        # Warp canvas layout grid to counteract drone yaw flight variance
        yaw_deg = np.rad2deg(uav_yaw)
        rotation_angle = -90.0 + yaw_deg
        h, w = img_uav_bgr.shape[:2]
        center = (w / 2.0, h / 2.0)
        M = cv2.getRotationMatrix2D(center, rotation_angle, 1.0)
        cos, sin = abs(M[0, 0]), abs(M[0, 1])
        new_w = int(h * sin + w * cos)
        new_h = int(h * cos + w * sin)
        M[0, 2] += (new_w - w) / 2.0
        M[1, 2] += (new_h - h) / 2.0
        img_uav_rot_bgr = cv2.warpAffine(img_uav_bgr, M, (new_w, new_h), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT)
        
        patch_bgr, r1, c1, uav_e_map, uav_n_map = self.extract_ortho_patch(float(row_meta["gps_longitude"]), float(row_meta["gps_latitude"]))
        if patch_bgr.size == 0:
            return None
            
        img1_rs, s1 = utils.resize_keep(img_uav_rot_bgr)
        img2_rs, s2 = utils.resize_keep(patch_bgr)
        img1_gray, img2_gray = utils.to_gray(img1_rs), utils.to_gray(img2_rs)
        
        # Execute individual feature tracking extractions
        kpts0, desc0 = self.run_superpoint(img1_gray)
        kpts1, desc1 = self.run_superpoint(img2_gray)
        
        if len(kpts0) == 0 or len(kpts1) == 0:
            return None
            
        # Match features via Brute-Force Matcher
        raw_matches = self.bf.match(desc0, desc1)
        if len(raw_matches) < config.MIN_GOOD_MATCHES:
            return None
            
        mkpts0 = np.array([kpts0[m.queryIdx] for m in raw_matches])
        mkpts1 = np.array([kpts1[m.trainIdx] for m in raw_matches])
        
        # Homography outlier rejection step
        src = mkpts0.reshape(-1, 1, 2)
        dst = mkpts1.reshape(-1, 1, 2)
        H, mask = cv2.findHomography(src, dst, cv2.RANSAC, config.RANSAC_REPROJ_THRESHOLD)
        if H is None:
            return None
            
        mask_bool = mask.ravel().astype(bool)
        mkpts0 = mkpts0[mask_bool]
        mkpts1 = mkpts1[mask_bool]
        good_matches = list(np.array(raw_matches)[mask_bool])
        
        # Project validation outputs back onto un-warped flight space perspectives
        mkpts0_full_rot = mkpts0 / s1
        H_inv = cv2.invertAffineTransform(M)
        mkpts0_original = cv2.transform(mkpts0_full_rot.reshape(-1, 1, 2), H_inv).reshape(-1, 2)
        mkpts1_full_patch = mkpts1 / s2
        
        self.save_matches_visualization(img1_rs, patch_bgr, img2_rs, mkpts0, mkpts1, uav_img_num)
        
        # Map pixels to coordinates using the DSM array
        obj_pts, img_pts = [], []
        with rasterio.open(config.DSM_PATH) as dsm:
            dsm_band = dsm.read(1)
            nodata = dsm.nodatavals[0]
            for i in range(len(mkpts0_original)):
                col_map = int(c1 + mkpts1_full_patch[i][0])
                row_map = int(r1 + mkpts1_full_patch[i][1])
                try:
                    e_map, n_map = self.px2utm_transform * (col_map, row_map)
                    r_dsm, c_dsm = dsm.index(e_map, n_map)
                    z = float(dsm_band[r_dsm, c_dsm])
                    if not np.isnan(z) and z != nodata:
                        obj_pts.append([e_map, n_map, z])
                        img_pts.append([mkpts0_original[i][0], mkpts0_original[i][1]])
                except:
                    continue
                    
        obj_pts = np.array(obj_pts, dtype=np.float64)
        img_pts = np.array(img_pts, dtype=np.float32)
        
        if len(obj_pts) < 6:
            return None
            
        camera_pos_map_expected = np.array([uav_e_map, uav_n_map, uav_alt_gps])
        obj_pts_local = (obj_pts - camera_pos_map_expected).astype(np.float32)
        
        # ------------------- Pipeline Track 1: Anchor PnP Solver -------------------
        rvec_prior, _ = cv2.Rodrigues(config.R_ANCHOR)
        retval, rvec_anchor, tvec_anchor, inliers_anchor = cv2.solvePnPRansac(
            obj_pts_local, img_pts, config.K.astype(np.float32), config.DIST_COEFFS.astype(np.float32),
            rvec=rvec_prior, tvec=np.zeros((3, 1)), useExtrinsicGuess=True,
            reprojectionError=3.0, iterationsCount=10000
        )
        if inliers_anchor is None or len(inliers_anchor) < 4:
            return None
            
        idx_a = inliers_anchor.ravel()
        _, rvec_anchor_ref, tvec_anchor_ref = cv2.solvePnP(
            obj_pts_local[idx_a], img_pts[idx_a], config.K.astype(np.float32), config.DIST_COEFFS.astype(np.float32),
            rvec=rvec_anchor, tvec=tvec_anchor, useExtrinsicGuess=True
        )
        R_map2cam_a, _ = cv2.Rodrigues(rvec_anchor_ref)
        R_cam2map_a = R_map2cam_a.T
        camera_pos_anchor = (-R_cam2map_a @ tvec_anchor_ref).reshape(3) + camera_pos_map_expected
        R_final_anchor = R_cam2map_a @ config.R_BORESIGHT
        v_ypr_anchor = utils.get_euler(R_final_anchor)
        mean_err_anchor, std_err_anchor = utils.compute_reprojection_error(
            obj_pts_local[idx_a], img_pts[idx_a], rvec_anchor_ref, tvec_anchor_ref,
            config.K.astype(np.float32), config.DIST_COEFFS.astype(np.float32)
        )
        
        # ------------------- Pipeline Track 2: Telemetry Prior PnP Solver -------------------
        R_imu = utils.R_from_quaternion(qw, qx, qy, qz)
        R_map2cam_prior = config.R_BORESIGHT @ R_imu
        rvec_guess, _ = cv2.Rodrigues(R_map2cam_prior)
        
        retval, rvec_prior_pnp, tvec_prior_pnp, inliers_prior = cv2.solvePnPRansac(
            obj_pts_local, img_pts, config.K.astype(np.float32), config.DIST_COEFFS.astype(np.float32),
            rvec=rvec_guess, tvec=np.zeros((3, 1)), useExtrinsicGuess=True,
            reprojectionError=3.0, iterationsCount=10000
        )
        if inliers_prior is None or len(inliers_prior) < 4:
            return None
            
        idx_p = inliers_prior.ravel()
        _, rvec_prior_ref, tvec_prior_ref = cv2.solvePnP(
            obj_pts_local[idx_p], img_pts[idx_p], config.K.astype(np.float32), config.DIST_COEFFS.astype(np.float32),
            rvec=rvec_prior_pnp, tvec=tvec_prior_pnp, useExtrinsicGuess=True
        )
        R_map2cam_p, _ = cv2.Rodrigues(rvec_prior_ref)
        R_cam2map_p = R_map2cam_p.T
        camera_pos_prior = (-R_cam2map_p @ tvec_prior_ref).reshape(3) + camera_pos_map_expected
        R_final_prior = R_cam2map_p @ config.R_BORESIGHT
        v_ypr_prior = utils.get_euler(R_final_prior)
        mean_err_prior, std_err_prior = utils.compute_reprojection_error(
            obj_pts_local[idx_p], img_pts[idx_p], rvec_prior_ref, tvec_prior_ref,
            config.K.astype(np.float32), config.DIST_COEFFS.astype(np.float32)
        )
        
        # Clear hardware cache at pipeline boundaries safely
        torch.cuda.empty_cache()
        
        return {
            "kp_uav": len(kpts0), "kp_patch": len(kpts1), "raw_matches": len(raw_matches), "good_matches": len(good_matches),
            "good_ratio": len(good_matches)/len(raw_matches) if len(raw_matches) > 0 else 0, "num_3d2d_corr": len(obj_pts_local),
            "pnp_inliers_anchor": len(inliers_anchor), "pnp_ratio_anchor": len(inliers_anchor)/len(obj_pts_local),
            "anchor_easting_est": camera_pos_anchor[0], "anchor_northing_est": camera_pos_anchor[1], "anchor_altitude_est": camera_pos_anchor[2],
            "anchor_easting_delta": camera_pos_anchor[0] - uav_e_map, "anchor_northing_delta": camera_pos_anchor[1] - uav_n_map, "anchor_altitude_delta": camera_pos_anchor[2] - uav_alt_gps,
            "anchor_yaw_est": v_ypr_anchor[0], "anchor_pitch_est": v_ypr_anchor[1], "anchor_roll_est": v_ypr_anchor[2],
            "anchor_yaw_delta": (v_ypr_anchor[0] - np.degrees(uav_yaw) + 180) % 360 - 180,
            "anchor_pitch_delta": (v_ypr_anchor[1] - np.degrees(float(row_meta["pitch"])) + 180) % 360 - 180,
            "anchor_roll_delta": (v_ypr_anchor[2] - np.degrees(float(row_meta["roll"])) + 180) % 360 - 180,
            "anchor_reproj_mean": mean_err_anchor, "anchor_reproj_std": std_err_anchor,
            "pnp_inliers_prior": len(inliers_prior), "pnp_ratio_prior": len(inliers_prior)/len(obj_pts_local),
            "prior_easting_est": camera_pos_prior[0], "prior_northing_est": camera_pos_prior[1], "prior_altitude_est": camera_pos_prior[2],
            "prior_easting_delta": camera_pos_prior[0] - uav_e_map, "prior_northing_delta": camera_pos_prior[1] - uav_n_map, "prior_altitude_delta": camera_pos_prior[2] - uav_alt_gps,
            "prior_yaw_est": v_ypr_prior[0], "prior_pitch_est": v_ypr_prior[1], "prior_roll_est": v_ypr_prior[2],
            "prior_yaw_delta": (v_ypr_prior[0] - np.degrees(uav_yaw) + 180) % 360 - 180,
            "prior_pitch_delta": (v_ypr_prior[1] - np.degrees(float(row_meta["pitch"])) + 180) % 360 - 180,
            "prior_roll_delta": (v_ypr_prior[2] - np.degrees(float(row_meta["roll"])) + 180) % 360 - 180,
            "prior_reproj_mean": mean_err_prior, "prior_reproj_std": std_err_prior,
            "gps_easting": uav_e_map, "gps_northing": uav_n_map, "gps_altitude": uav_alt_gps,
            "imu_yaw": np.degrees(uav_yaw), "imu_pitch": np.degrees(float(row_meta["pitch"])), "imu_roll": np.degrees(float(row_meta["roll"]))
        }