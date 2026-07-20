import os
import glob
import pandas as pd
import config
from pipeline import SuperGluePosePipeline

def main():
    image_paths = sorted(glob.glob(os.path.join(config.IMAGE_DIR, "frame_*.png")))
    
    if not os.path.exists(config.META_PATH):
        raise FileNotFoundError(f"Metadata file index missing at: {config.META_PATH}")
        
    meta_data = pd.read_csv(config.META_PATH)
    results = []
    
    pipeline = SuperGluePosePipeline()

    print("Beginning SuperPoint + SuperGlue deep match pose pipeline tracking loops...")
    for image_path in image_paths:
        filename = os.path.basename(image_path)
        uav_img_num = int(filename.split("_")[1].split(".")[0])
        
        # Enforce baseline constraint parameters
        if uav_img_num < 214:
            continue
            
        if not (0 <= uav_img_num < len(meta_data)):
            print(f"Skipping index overflow frame: {uav_img_num}")
            continue
            
        print(f"Processing Frame Sequence Index: {uav_img_num} -> {filename}")
        row_meta = meta_data.iloc[uav_img_num]
        
        try:
            metrics = pipeline.process_frame(image_path, row_meta, uav_img_num)
            if metrics is not None:
                metrics["image_id"] = uav_img_num
                metrics["filename"] = filename
                results.append(metrics)
            else:
                print(f"Frame validation checks failed or fell short for image: {filename}")
        except Exception as e:
            print(f"Error parsing tracking operations on frame {uav_img_num}: {str(e)}")

    if results:
        results_df = pd.DataFrame(results)
        output_csv = "uav_pose_superglue.csv"
        results_df.to_csv(output_csv, index=False)
        print(f"Successfully compiled match tracking telemetry matrix to: {output_csv}")
    else:
        print("No valid orientations or metrics parsed for this pipeline run step.")

if __name__ == "__main__":
    main()