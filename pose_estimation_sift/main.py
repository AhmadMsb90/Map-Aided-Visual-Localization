import os
import glob
import pandas as pd
import config
from pipeline import PoseEstimationPipeline

def main():
    # Setup paths and pipeline variables
    image_paths = sorted(glob.glob(os.path.join(config.IMAGE_DIR, "frame_*.png")))
    
    if not os.path.exists(config.META_PATH):
        raise FileNotFoundError(f"Metadata file missing: {config.META_PATH}")
        
    meta_data = pd.read_csv(config.META_PATH)
    
    # Instantiate the pre-warmed processing pipeline
    pipeline = PoseEstimationPipeline()
    results = []

    print(f"Beginning spatial localization for frame collection...")
    for image_path in image_paths:
        filename = os.path.basename(image_path)
        uav_img_num = int(filename.split("_")[1].split(".")[0])
        
        # Specific structural filtering constraints
        if uav_img_num < 214:
            continue
            
        if not (0 <= uav_img_num < len(meta_data)):
            print(f"Index bounds check out of range for frame index {uav_img_num}. Skipping.")
            continue
            
        row_meta = meta_data.iloc[uav_img_num]
        print(f"Processing Frame: {uav_img_num} -> {filename}")
        
        try:
            frame_metrics = pipeline.process_frame(image_path, row_meta)
            if frame_metrics is not None:
                frame_metrics["image_id"] = uav_img_num
                frame_metrics["filename"] = filename
                results.append(frame_metrics)
            else:
                print(f"Frame validation or optimization failed for: {filename}")
        except Exception as e:
            print(f"Error parsing localization computation sequence on {filename}: {str(e)}")

    # Compile dataset export structures
    if results:
        results_df = pd.DataFrame(results)
        output_name = "uav_pose_results_sift.csv"
        results_df.to_csv(output_name, index=False)
        print(f"Successfully exported localization metrics data records into: {output_name}")
    else:
        print("No dynamic matching outputs achieved across current collection profiles.")

if __name__ == "__main__":
    main()