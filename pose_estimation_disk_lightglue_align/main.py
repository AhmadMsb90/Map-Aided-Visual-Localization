
import os
import glob
import pandas as pd
import config
from pipeline import DeepPosePipeline

def main():
    # Gather flight imagery frame pathways
    image_paths = sorted(glob.glob(os.path.join(config.IMAGE_DIR, "frame_*.png")))
    
    if not os.path.exists(config.META_PATH):
        raise FileNotFoundError(f"Metadata index missing at: {config.META_PATH}")
        
    meta_data = pd.read_csv(config.META_PATH)
    results = []

    # Instantiate the pre-compiled tensor-matching pipeline
    pipeline = DeepPosePipeline()

    print("Initiating DISK + LightGlue dynamic camera tracking...")
    for image_path in image_paths:
        filename = os.path.basename(image_path)
        uav_imag_num = int(filename.split("_")[1].split(".")[0])
        
        # Operational constraints filter
        if uav_imag_num < 214:
            continue
            
        if not (0 <= uav_imag_num < len(meta_data)):
            print(f"Index mapping overflow for frame {uav_imag_num}. Skipping.")
            continue
            
        print(f"\nProcessing frame reference: {uav_imag_num} ({filename})")
        row_meta = meta_data.iloc[uav_imag_num]
        
        try:
            metrics = pipeline.process_frame(image_path, row_meta, uav_imag_num)
            if metrics is not None:
                metrics["image_id"] = uav_imag_num
                metrics["filename"] = filename
                results.append(metrics)
            else:
                print(f"Frame matching failed criteria on: {filename}")
        except Exception as e:
            print(f"Exception encountered during frame {uav_imag_num} processing: {str(e)}")

    # Serialize complete session data metrics to disk
    if results:
        results_df = pd.DataFrame(results)
        output_csv = "uav_pose_disk_lightglue.csv"
        results_df.to_csv(output_csv, index=False)
        print(f"Saved match calculation matrix records to: {output_csv}")
    else:
        print("No valid spatial positions resolved during this iteration batch pass.")

if __name__ == "__main__":
    main()
