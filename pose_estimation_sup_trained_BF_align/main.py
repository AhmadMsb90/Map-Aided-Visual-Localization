import os
import glob
import pandas as pd
import config
from pipeline import SuperPointBFPipeline

def main():
    image_paths = sorted(glob.glob(os.path.join(config.IMAGE_DIR, "frame_*.png")))
    
    if not os.path.exists(config.META_PATH):
        raise FileNotFoundError(f"Metadata file path missing at: {config.META_PATH}")
        
    meta_data = pd.read_csv(config.META_PATH)
    results = []
    
    pipeline = SuperPointBFPipeline()

    print("Executing SuperPoint + BF Matcher camera tracking iteration batches...")
    for image_path in image_paths:
        filename = os.path.basename(image_path)
        uav_img_num = int(filename.split("_")[1].split(".")[0])
        
        # Apply operational frame constraints
        if uav_img_num < 214:
            continue
            
        if not (0 <= uav_img_num < len(meta_data)):
            print(f"Index target error boundary skip on frame {uav_img_num}.")
            continue
            
        print(f"Processing Frame Sequence Reference: {uav_img_num} -> {filename}")
        row_meta = meta_data.iloc[uav_img_num]
        
        try:
            metrics = pipeline.process_frame(image_path, row_meta, uav_img_num)
            if metrics is not None:
                metrics["image_id"] = uav_img_num
                metrics["filename"] = filename
                results.append(metrics)
            else:
                print(f"Tracking filters rejected match resolution step on: {filename}")
        except Exception as e:
            print(f"Error parsing localization frame tracking pipelines on image {uav_img_num}: {str(e)}")

    if results:
        results_df = pd.DataFrame(results)
        output_csv = "uav_pose_superpoint_bf.csv"
        results_df.to_csv(output_csv, index=False)
        print(f"Successfully generated dynamic flight matching matrices tracking profile: {output_csv}")
    else:
        print("No valid orientations tracked across current frame boundaries profile.")

if __name__ == "__main__":
    main()