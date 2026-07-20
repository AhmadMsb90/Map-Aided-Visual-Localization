import os
import pandas as pd

CSV = "metadata_with_euler.csv"
KEEP = 1054

# Read CSV (auto-detect delimiter)
df = pd.read_csv(CSV, engine="python")

# Keep last 1054 rows
df = df.iloc[-KEEP:].reset_index(drop=True)

# Rename images + update CSV
for i in range(KEEP):
    old_name = df.at[i, "filename"]
    new_name = f"{i+1:05d}.png"

    if not os.path.isfile(old_name):
        raise FileNotFoundError(f"Image not found: {old_name}")

    os.rename(old_name, new_name)
    df.at[i, "filename"] = new_name

# Remove old frame_*.png files
for f in os.listdir("."):
    if f.startswith("frame_") and f.endswith(".png"):
        os.remove(f)

# Write updated CSV
df.to_csv(CSV, index=False)

print("Done. 1054 images + CSV aligned.")
