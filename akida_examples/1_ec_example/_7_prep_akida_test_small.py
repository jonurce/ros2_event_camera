# _7_prep_akida_test_small.py
# Exact replication of BrainChip Akida gaze example preprocessing
# → Input:  [N, 2, 50, 96, 128]   (pos/neg binary, 500ms)
# → Output: [N, 3, 50, 3, 4]       (target heatmap: confidence + x/y offset)
# → Only keeps open-eye samples (state == 0)

# This version creates a smaller test set for inference in Raspberry Pi


import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent / "training"))


import torch
from pathlib import Path
import re
from tqdm import tqdm

from training._1_4_preprocess_akida import preprocess_recording

# ========================================
# CONFIG — Akida-exact parameters
# ========================================
T_LABELED = 10000                  # 10 ms per bin
N_BINS = 50                    # 50 bins = 500 ms
H_ORIG, W_ORIG = 480, 640
H_DOWN, W_DOWN = 96, 128       # Downsampled resolution (5× smaller)
H_OUT, W_OUT = 3, 4            # Final feature map size after 32× downsampling

DATA_ROOT = Path("/home/dronelab-pc-1/Jon/IndustrialProject/akida_examples/1_ec_example/training/event-based-eye-tracking-cvpr-2025/3ET+ dataset/event_data")
OUTPUT_ROOT = Path("/home/dronelab-pc-1/Jon/IndustrialProject/akida_examples/1_ec_example/training/preprocessed_akida_test_small")
OUTPUT_ROOT.mkdir(exist_ok=True)

MAX_FOLDERS = 2                  # Limit number of folders to process (for small test set)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Preprocessing for Akida-exact model (T=50, 128×96, target heatmaps)")





# ========================================
# 5. MAIN
# ========================================
if __name__ == "__main__":
    print("Starting Small-Test-Akida-exact preprocessing (T=50, 128×96, target heatmaps)")
    total = 0
    split = 'test'
    split_path = DATA_ROOT / split

    if not split_path.exists():
        print(f"Split {split} not found!")
        exit()

    folders = sorted([f for f in split_path.iterdir() if f.is_dir()])
    print(f"\nProcessing {split.upper()} — {MAX_FOLDERS} recordings")

    for i, folder in enumerate(tqdm(folders, total=MAX_FOLDERS)):
        if i >= MAX_FOLDERS:
            break
        n = preprocess_recording(folder, split, OUTPUT_ROOT)
        if n > 0:
            total += n
            tqdm.write(f"  {folder.name}: {n} samples → voxels.pt + heatmaps.pt")

    print(f"\nDONE! Total open-eye samples: {total}")
    print(f"Output folder: {OUTPUT_ROOT}")