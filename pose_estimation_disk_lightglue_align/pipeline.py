import os
import sys
import cv2
import numpy as np
import rasterio
from rasterio.transform import rowcol
from pyproj import Transformer
import torch
import config
import utils

# Perform sys.path insertion before importing local lightglue components 
if str(config.SG_ROOT) not in sys.path:
    sys.path.insert(0, str(config.SG_ROOT))  # 

from lightglue.lightglue import LightGlue  # 
from lightglue.disk import DISK            # 

class DeepPosePipeline:
    def __init__(self):
        # Verify base geographical raster asset dependencies
        if not os.path.exists(config.ORTHO_PATH):
            raise FileNotFoundError(f"Orthomosaic file not found: {config.ORTHO_PATH}")  
            
        # Parse geographic spatial transforms once during instantiation to reduce file read overhead
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
        
        # Initialize transformation metrics and network entities
        self.transformer = Transformer.from_crs("EPSG:4326", self.ortho_crs, always_xy=True) 
        
        self.disk = DISK(
            detection_threshold=config.DETECTION_THRESHOLD,
            max_num_keypoints=config.MAX_NUM_KEYPOINTS
        ).to(config.DEVICE).eval()  
        
        self.matcher = LightGlue(
            features="disk",
            pruning_keypoint_thresholds=config.PRUNING_KEYPOINT_THRESH
        ).to(config.DEVICE).eval()  

    def extract_ortho_patch(self, lon, lat):
        """Isolate a spatial coordinate matching crop patch from the orthomosaic."""
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

    def save_matches_visualization(self, img1_rs, patch_bgr, img2_rs, s1, mkpts0, mkpts1, uav_imag_num):
        """Draw match lines side-by-side and output the inspection canvas to storage."""
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
        save_path = os.path.join(config.VIS_DIR, f"match_{uav_imag_num}.png")  
        cv2.imwrite(save_path, canvas)  
        print("Saved match image:", save_path)  

    def process_frame(self, image_path, row_meta, uav_imag_num):
        """Execute deep feature matching, spatial telemetry fusion, and multi-hypothesis PnP checks."""
        # Parse flight vector strings
        uav_alt_gps = float(row_meta["gps_altitude"])       
        uav_yaw = float(row_meta["yaw"])                    
        qx, qy, qz, qw = float(row_meta["imu_orientation_x"]), float(row_meta["imu_orientation_y"]), float(row_meta["imu_orientation_z"]), float(row_meta["imu_orientation_w"])  
        
        img_uav_bgr = cv2.imread(image_path)  
        if img_uav_bgr is None:  
            print("Image read failed")       
            return None
        img_uav_bgr = img_uav_bgr.astype("uint8")  
        
        # Compensate for flight yaw anomalies by pre-rotating input images
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
        img_uav_rot_bgr = cv2.warpAffine(
            img_uav_bgr, M, (new_w, new_h),
            flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT
        )  
        
        # Pull matching geographical reference context maps
        patch_bgr, r1, c1, uav_e_map, uav_n_map = self.extract_ortho_patch(float(row_meta["gps_longitude"]), float(row_meta["gps_latitude"]))  
        if patch_bgr.size == 0:  
            print("Patch extraction failed")  
            return None
            
        img1_rs, s1 = utils.resize_keep(img_uav_rot_bgr) 
        img2_rs, s2 = utils.resize_keep(patch_bgr)       
        img1_gray, img2_gray = utils.to_gray(img1_rs), utils.to_gray(img2_rs) 
        
        # Build Torch neural network input arrays
        img0_tensor = torch.from_numpy(img1_gray.astype(np.float32) / 255.0)[None, None].to(config.DEVICE) 
        img1_tensor = torch.from_numpy(img2_gray.astype(np.float32) / 255.0)[None, None].to(config.DEVICE) 
        
        with torch.no_grad(): 
            feats0 = self.disk({'image': img0_tensor}) 
            feats1 = self.disk({'image': img1_tensor})  
            match_input = {
                'image0': img0_tensor,
                'keypoints0': feats0['keypoints'],
                'descriptors0': feats0['descriptors'],
                'image1': img1_tensor,
                'keypoints1': feats1['keypoints'],
                'descriptors1': feats1['descriptors'],
            }
            match_output = self.matcher(match_input)  
            
        kpts0 = feats0['keypoints'][0].cpu().numpy()  
        kpts1 = feats1['keypoints'][0].cpu().numpy()  
        matches0 = match_output['matches0'][0].cpu().numpy()  
        
        valid = matches0 != -1  
        mkpts0 = kpts0[valid]  
        mkpts1 = kpts1[matches0[valid]]  
        
        good_matches = len(mkpts0)  
        print("LightGlue matches:", good_matches)  
        if good_matches < 20:  
            print("Too few matches")  
            return None
            
        # Homography geometric validation filters
        src = mkpts0.reshape(-1, 1, 2)  
        dst = mkpts1.reshape(-1, 1, 2)  
        H, mask = cv2.findHomography(src, dst, cv2.RANSAC, 5.0)  
        if H is None:  
            print("Homography failed")  
            return None
        mask_bool = mask.ravel().astype(bool)  
        mkpts0 = mkpts0[mask_bool]  
        mkpts1 = mkpts1[mask_bool]  
        
        # Project validation outputs into scaled imagery context
        mkpts0_full_rot = mkpts0 / s1  
        H_inv = cv2.invertAffineTransform(M)  
        mkpts0_original = cv2.transform(mkpts0_full_rot.reshape(-1, 1, 2), H_inv).reshape(-1, 2)  
        
        # Save structural debugging assets to disk
        self.save_matches_visualization(img1_rs, patch_bgr, img2_rs, s1, mkpts0, mkpts1, uav_imag_num)
        
        if len(mkpts0) >= 4:  
            src_pts = mkpts0.reshape(-1, 1, 2)  
            dst_pts = mkpts1.reshape(-1, 1, 2)  
            _, mask_h = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 3.0)  
            maskH = mask_h.ravel().astype(bool)  
            homography_inliers = int(maskH.sum())  
            homography_ratio = homography_inliers / len(mkpts0)  
        else:
            maskH = None  
            homography_inliers = 0  
            homography_ratio = 0.0  
            
        # Elevate coordinates using DSM values
        obj_pts, img_pts = [], []  
        mkpts1_full_patch = mkpts1 / s2  
        
        with rasterio.open(config.DSM_PATH) as dsm:  
            dsm_band = dsm.read(1)  
            nodata = dsm.nodatavals[0]  
            for i in range(len(mkpts0_original)):  
                col_patch_full = mkpts1_full_patch[i][0]  
                row_patch_full = mkpts1_full_patch[i][1]  
                col_map = int(c1 + col_patch_full)  
                row_map = int(r1 + row_patch_full)  
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
            print("Too few 3D points")  
            return None
            
        camera_pos_map_expected = np.array([uav_e_map, uav_n_map, uav_alt_gps])  
        obj_pts_local = (obj_pts - camera_pos_map_expected).astype(np.float32)  
        
        # ------------------- Hypothesis 1: Anchor PnP Solver -------------------
        rvec_prior, _ = cv2.Rodrigues(config.R_ANCHOR)  
        retval, rvec_anchor, tvec_anchor, inliers_anchor = cv2.solvePnPRansac(
            obj_pts_local, img_pts, config.K.astype(np.float32), config.DIST_COEFFS.astype(np.float32),
            rvec=rvec_prior, tvec=np.zeros((3, 1)), useExtrinsicGuess=True,
            reprojectionError=3.0, iterationsCount=10000
        )  
        
        if inliers_anchor is None:  
            print("Anchor PnP failed")  
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
        
        # ------------------- Hypothesis 2: Telemetry Prior PnP Solver -------------------
        R_imu = utils.R_from_quaternion(qw, qx, qy, qz)  
        R_map2cam_prior = config.R_BORESIGHT @ R_imu  
        rvec_guess, _ = cv2.Rodrigues(R_map2cam_prior)  
        
        retval, rvec_prior_pnp, tvec_prior_pnp, inliers_prior = cv2.solvePnPRansac(
            obj_pts_local, img_pts, config.K.astype(np.float32), config.DIST_COEFFS.astype(np.float32),
            rvec=rvec_guess, tvec=np.zeros((3, 1)), useExtrinsicGuess=True,
            reprojectionError=3.0, iterationsCount=10000
        )  
        
        if inliers_prior is None:  
            print("Prior PnP failed")  
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
        
        # Pack performance telemetry statistics
        output_metrics = {
            "kp_uav": len(kpts0), "kp_patch": len(kpts1), "raw_matches": len(matches0), "good_matches": len(mkpts0),  
            "good_ratio": len(mkpts0)/len(matches0) if len(matches0) > 0 else 0,  
            "homography_inliers": int(maskH.sum()) if maskH is not None else 0,  
            "homography_ratio": (int(maskH.sum())/len(mkpts0)) if (maskH is not None and len(mkpts0) > 0) else 0.0,  
            "num_3d2d_corr": len(obj_pts_local),  
            "pnp_inliers_anchor": len(inliers_anchor) if inliers_anchor is not None else 0,  
            "pnp_ratio_anchor": (len(inliers_anchor)/len(obj_pts_local)) if (inliers_anchor is not None and len(obj_pts_local) > 0) else 0,  
            "anchor_easting_est": camera_pos_anchor[0], "anchor_northing_est": camera_pos_anchor[1], "anchor_altitude_est": camera_pos_anchor[2],  
            "anchor_easting_delta": camera_pos_anchor[0] - uav_e_map, "anchor_northing_delta": camera_pos_anchor[1] - uav_n_map, "anchor_altitude_delta": camera_pos_anchor[2] - uav_alt_gps,  
            "anchor_yaw_est": v_ypr_anchor[0], "anchor_pitch_est": v_ypr_anchor[1], "anchor_roll_est": v_ypr_anchor[2],  
            "anchor_yaw_delta": (v_ypr_anchor[0] - np.degrees(uav_yaw) + 180) % 360 - 180,  
            "anchor_pitch_delta": (v_ypr_anchor[1] - np.degrees(float(row_meta["pitch"])) + 180) % 360 - 180,  
            "anchor_roll_delta": (v_ypr_anchor[2] - np.degrees(float(row_meta["roll"])) + 180) % 360 - 180,  
            "anchor_reproj_mean": mean_err_anchor, "anchor_reproj_std": std_err_anchor,  
            "pnp_inliers_prior": len(inliers_prior) if inliers_prior is not None else 0,  
            "pnp_ratio_prior": (len(inliers_prior)/len(obj_pts_local)) if (inliers_prior is not None and len(obj_pts_local) > 0) else 0,  
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
        
        # Garbage collection step to clear GPU memory cache safely outside processing loops 
        del img0_tensor, img1_tensor, feats0, feats1, match_output  
        torch.cuda.empty_cache()  
        
        return output_metrics
