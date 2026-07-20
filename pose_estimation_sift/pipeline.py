import os
import cv2
import numpy as np
import rasterio
from rasterio.transform import rowcol
from pyproj import Transformer
import config
import utils

class PoseEstimationPipeline:
    def __init__(self):
        # Validate spatial map references
        if not os.path.exists(config.ORTHO_PATH):
            raise FileNotFoundError(f"Orthomosaic map file not found: {config.ORTHO_PATH}")
        
        # Load orthomosaic properties globally to manage throughput overhead
        with rasterio.open(config.ORTHO_PATH) as ds:
            self.image_map_raw = ds.read(out_dtype="uint8")
            self.px2utm_transform = ds.transform
            self.ortho_crs = ds.crs
            self.bounds = ds.bounds
        
        # Determine base resolution limits and structural scale variations
        self.GSD_x = abs(self.px2utm_transform.a)
        self.GSD_y = abs(self.px2utm_transform.e)
        
        # Preprocess background array structural dimensions
        if self.image_map_raw.shape[0] > 3:
            self.image_map_raw = self.image_map_raw[:3, :, :]
        
        image_map_rgb = np.transpose(self.image_map_raw, (1, 2, 0))
        self.image_map_bgr = cv2.cvtColor(image_map_rgb, cv2.COLOR_RGB2BGR)
        self.h_map, self.w_map = self.image_map_bgr.shape[:2]
        
        # Instantiating coordinate reference projection configurations
        self.transformer = Transformer.from_crs("EPSG:4326", self.ortho_crs, always_xy=True)
        self.sift = cv2.SIFT_create(nfeatures=config.MAX_FEATURES)
        self.bf = cv2.BFMatcher(cv2.NORM_L2)

    def extract_ortho_patch(self, lon, lat):
        """Extract a bounding region from the master orthomosaic array."""
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

    def process_frame(self, image_path, row_meta):
        """Execute cross-image geometric verification, SIFT matching, and coordinate reconstruction."""
        # Unpack telemetry vectors
        uav_alt_gps = float(row_meta["gps_altitude"])
        uav_yaw, uav_pitch, uav_roll = float(row_meta["yaw"]), float(row_meta["pitch"]), float(row_meta["roll"])
        qx, qy, qz, qw = float(row_meta["imu_orientation_x"]), float(row_meta["imu_orientation_y"]), float(row_meta["imu_orientation_z"]), float(row_meta["imu_orientation_w"])
        
        img_uav_bgr = cv2.imread(image_path)
        if img_uav_bgr is None:
            return None
        img_uav_bgr = img_uav_bgr.astype("uint8")
        
        # Spatial contextual scaling
        patch_bgr, r1, c1, uav_e_map, uav_n_map = self.extract_ortho_patch(float(row_meta["gps_longitude"]), float(row_meta["gps_latitude"]))
        if patch_bgr.size == 0:
            return None
            
        img1_rs, s1 = utils.resize_keep(img_uav_bgr)
        img2_rs, s2 = utils.resize_keep(patch_bgr)
        
        img1_gray, img2_gray = utils.to_gray(img1_rs), utils.to_gray(img2_rs)
        
        # Structural Feature Extraction
        k1, d1 = self.sift.detectAndCompute(img1_gray, None)
        k2, d2 = self.sift.detectAndCompute(img2_gray, None)
        if d1 is None or d2 is None:
            return None
            
        # Feature Matching via Nearest Neighbors Ratio Test
        good = []
        for m, n in self.bf.knnMatch(d1, d2, k=2):
            if m.distance < config.RATIO_TEST * n.distance:
                good.append(m)
                
        if len(good) < config.MIN_GOOD_MATCHES:
            return None
            
        src = np.float32([k1[m.queryIdx].pt for m in good]).reshape(-1, 1, 2)
        dst = np.float32([k2[m.trainIdx].pt for m in good]).reshape(-1, 1, 2)
        
        # Outlier Homography Rejection Filter
        H, mask = cv2.findHomography(src, dst, cv2.RANSAC, config.RANSAC_REPROJ_THRESHOLD)
        if H is None:
            return None
        maskH = mask.ravel().astype(bool)
        good = [good[i] for i in range(len(good)) if maskH[i]]
        
        # Dem-Elevated Elevation Mapping
        obj_pts, img_pts = [], []
        with rasterio.open(config.DSM_PATH) as dsm:
            dsm_band = dsm.read(1)
            nodata = dsm.nodatavals[0]
            for rec in good:
                u_pt = k1[rec.queryIdx].pt
                col = int(c1 + k2[rec.trainIdx].pt[0] / s2)
                row_ = int(r1 + k2[rec.trainIdx].pt[1] / s2)
                try:
                    e_map, n_map = self.px2utm_transform * (col, row_)
                    r_dsm, c_dsm = dsm.index(e_map, n_map)
                    z = float(dsm_band[r_dsm, c_dsm])
                    if not np.isnan(z) and z != nodata:
                        obj_pts.append([e_map, n_map, z])
                        img_pts.append([u_pt[0] / s1, u_pt[1] / s1])
                except:
                    continue
                    
        obj_pts = np.array(obj_pts, dtype=np.float64)
        img_pts = np.array(img_pts, dtype=np.float32)
        if len(obj_pts) < 4:
            return None
            
        camera_pos_map_expected = np.array([uav_e_map, uav_n_map, uav_alt_gps])
        obj_pts_local = (obj_pts - camera_pos_map_expected).astype(np.float32)
        
        # ------------------- Anchor PnP Pipeline Method -------------------
        rvec_prior, _ = cv2.Rodrigues(config.R_ANCHOR)
        retval, rvec_anchor, tvec_anchor, inliers_anchor = cv2.solvePnPRansac(
            obj_pts_local, img_pts, config.K.astype(np.float32), config.DIST_COEFFS.astype(np.float32),
            rvec=rvec_prior, tvec=np.zeros((3, 1)), useExtrinsicGuess=True,
            reprojectionError=3.0, iterationsCount=10000, flags=cv2.SOLVEPNP_ITERATIVE
        )
        if inliers_anchor is None or len(inliers_anchor) < 4:
            return None
            
        ok, rvec_anchor_ref, tvec_anchor_ref = cv2.solvePnP(
            obj_pts_local[inliers_anchor.ravel()], img_pts[inliers_anchor.ravel()],
            config.K.astype(np.float32), config.DIST_COEFFS.astype(np.float32),
            rvec=rvec_anchor, tvec=tvec_anchor, useExtrinsicGuess=True
        )
        R_map2cam, _ = cv2.Rodrigues(rvec_anchor_ref)
        mean_err_anchor, std_err_anchor = utils.compute_reprojection_error(
            obj_pts_local[inliers_anchor.ravel()], img_pts[inliers_anchor.ravel()],
            rvec_anchor_ref, tvec_anchor_ref, config.K.astype(np.float32), config.DIST_COEFFS.astype(np.float32)
        )
        R_cam2map = R_map2cam.T
        camera_pos_anchor = (-R_cam2map @ tvec_anchor_ref).reshape(3) + camera_pos_map_expected
        R_final_anchor = R_cam2map @ config.R_BORESIGHT
        v_ypr_anchor = utils.get_euler(R_final_anchor)
        
        # ------------------- Prior PnP Pipeline Method -------------------
        R_imu = utils.R_from_quaternion(qw, qx, qy, qz)
        R_map2cam_prior = config.R_BORESIGHT @ R_imu
        rvec_guess, _ = cv2.Rodrigues(R_map2cam_prior)
        
        retval, rvec_prior_pnp, tvec_prior_pnp, inliers_prior = cv2.solvePnPRansac(
            obj_pts_local, img_pts, config.K.astype(np.float32), config.DIST_COEFFS.astype(np.float32),
            rvec=rvec_guess, tvec=np.zeros((3, 1)), useExtrinsicGuess=True,
            reprojectionError=3.0, iterationsCount=10000, flags=cv2.SOLVEPNP_ITERATIVE
        )
        if inliers_prior is None or len(inliers_prior) < 4:
            return None
            
        idx = inliers_prior.ravel()
        ok, rvec_prior_ref, tvec_prior_ref = cv2.solvePnP(
            obj_pts_local[idx], img_pts[idx], config.K.astype(np.float32), config.DIST_COEFFS.astype(np.float32),
            rvec=rvec_prior_pnp, tvec=tvec_prior_pnp, useExtrinsicGuess=True
        )
        R_map2cam_p, _ = cv2.Rodrigues(rvec_prior_ref)
        R_cam2map_p = R_map2cam_p.T
        camera_pos_prior = (-R_cam2map_p @ tvec_prior_ref).reshape(3) + camera_pos_map_expected
        R_final_prior = R_cam2map_p @ config.R_BORESIGHT
        v_ypr_prior = utils.get_euler(R_final_prior)
        mean_err_prior, std_err_prior = utils.compute_reprojection_error(
            obj_pts_local[idx], img_pts[idx], rvec_prior_ref, tvec_prior_ref,
            config.K.astype(np.float32), config.DIST_COEFFS.astype(np.float32)
        )
        
        # Package and return compiled performance stats
        return {
            "kp_uav": len(k1), "kp_patch": len(k2), "raw_matches": len(src), "good_matches": len(good),
            "good_ratio": len(good)/len(src) if len(src) > 0 else 0, "num_3d2d_corr": len(obj_pts_local),
            "pnp_inliers_anchor": len(inliers_anchor), "anchor_easting_est": camera_pos_anchor[0],
            "anchor_northing_est": camera_pos_anchor[1], "anchor_altitude_est": camera_pos_anchor[2],
            "anchor_easting_delta": camera_pos_anchor[0] - uav_e_map, "anchor_northing_delta": camera_pos_anchor[1] - uav_n_map,
            "anchor_altitude_delta": camera_pos_anchor[2] - uav_alt_gps, "anchor_yaw_est": v_ypr_anchor[0],
            "anchor_pitch_est": v_ypr_anchor[1], "anchor_roll_est": v_ypr_anchor[2],
            "anchor_yaw_delta": (v_ypr_anchor[0] - np.degrees(uav_yaw) + 180) % 360 - 180,
            "anchor_pitch_delta": (v_ypr_anchor[1] - np.degrees(uav_pitch) + 180) % 360 - 180,
            "anchor_roll_delta": (v_ypr_anchor[2] - np.degrees(uav_roll) + 180) % 360 - 180,
            "reproj_error_anchor_mean": mean_err_anchor, "reproj_error_anchor_std": std_err_anchor,
            "pnp_inliers_prior": len(inliers_prior), "pnp_ratio_prior": len(inliers_prior)/len(obj_pts_local),
            "pnp_ratio_anchor": len(inliers_anchor)/len(obj_pts_local),
            "prior_easting_est": camera_pos_prior[0], "prior_northing_est": camera_pos_prior[1], "prior_altitude_est": camera_pos_prior[2],
            "prior_easting_delta": camera_pos_prior[0] - uav_e_map, "prior_northing_delta": camera_pos_prior[1] - uav_n_map,
            "prior_altitude_delta": camera_pos_prior[2] - uav_alt_gps, "prior_yaw_est": v_ypr_prior[0],
            "prior_pitch_est": v_ypr_prior[1], "prior_roll_est": v_ypr_prior[2],
            "prior_yaw_delta": (v_ypr_prior[0] - np.degrees(uav_yaw) + 180) % 360 - 180,
            "prior_pitch_delta": (v_ypr_prior[1] - np.degrees(uav_pitch) + 180) % 360 - 180,
            "prior_roll_delta": (v_ypr_prior[2] - np.degrees(uav_roll) + 180) % 360 - 180,
            "reproj_error_prior_mean": mean_err_prior, "reproj_error_prior_std": std_err_prior,
            "gps_easting": uav_e_map, "gps_northing": uav_n_map, "gps_altitude": uav_alt_gps,
            "imu_yaw": np.degrees(uav_yaw), "imu_pitch": np.degrees(uav_pitch), "imu_roll": np.degrees(uav_roll)
        }