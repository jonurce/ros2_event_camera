# 1_preprocess.py — MEMORY-SAFE VERSION
# → Saves one .npz per recording
# → Total RAM: < 2 GB

import h5py
import numpy as np
from pathlib import Path
import re
from tqdm import tqdm

# ========================================
# CONFIG
# ========================================
T_BINS = 10
BIN_MS = 10
WINDOW_MS = T_BINS * BIN_MS
H, W = 480, 640
DATA_ROOT = Path("/home/jetson/Jon/IndustrialProject/akida_examples/1_ec_example/training/event-based-eye-tracking-cvpr-2025/3ET+ dataset/event_data")
OUTPUT_DIR = Path("/home/jetson/Jon/IndustrialProject/akida_examples/1_ec_example/training/preprocessed")
OUTPUT_DIR.mkdir(exist_ok=True)

# ========================================
# 1. LABEL LOADER
# ========================================
def load_labels(txt_path):
    with open(txt_path, 'r') as f:
        lines = f.readlines()
    labels = []
    for line in lines:
        nums = re.findall(r'\d+', line)
        if len(nums) >= 3:
            x, y, state = map(int, nums[:3])
            labels.append([x, y, state])
    return np.array(labels, dtype=np.int32)

# ========================================
# 2. HARD BINNING VOXEL
# ========================================
def events_to_voxel_grid(events, t_start, t_end):
    """Hard binning — 100× faster than histogramdd"""
    voxel = np.zeros((T_BINS, H, W), dtype=np.float32)

    # Direct array access — no Python objects
    t = events['t']
    x = events['x']
    y = events['y']
    p = events['p']

    # Time → bin (float → int)
    dt = (t_end - t_start + 1e-6)
    bin_idx = ((t - t_start) * T_BINS / dt).astype(np.int32)
    bin_idx = np.clip(bin_idx, 0, T_BINS - 1)

    # Polarity: +1 or -1
    pol = np.where(p == 1, 1.0, -1.0)

    # ONE LINE — pure C loop
    np.add.at(voxel, (bin_idx, y, x), pol)

    return voxel  # (10, 480, 640)

# ========================================
# 3. PREPROCESS ONE RECORDING
# ========================================
import time

def preprocess_recording(folder, split):
    h5_file = folder / f"{folder.name}.h5"
    txt_file = folder / "label.txt"
    if not h5_file.exists() or not txt_file.exists():
        return 0

    t0 = time.time()
    with h5py.File(h5_file, 'r') as f:
        events = f['events'][()]
    t_load = time.time() - t0

    t0 = time.time()
    labels = load_labels(txt_file)
    t_labels = time.time() - t0

    if len(labels) == 0 or len(events) == 0:
        return 0

    t_events = events['t']
    t_labels_us = np.arange(len(labels)) * 10000

    voxels = []
    labels_out = []
    t_voxel_total = 0

    print(f"  → {folder.name}: {len(events)} events, {len(labels)} labels", end="")

    for i in range(len(labels)):
        t_label = t_labels_us[i]
        t_start = t_label - WINDOW_MS * 1000
        t_end = t_label

        t0 = time.time()
        mask = (t_events >= t_start) & (t_events <= t_end)
        window_events = events[mask]
        t_mask = time.time() - t0

        if len(window_events) == 0:
            continue

        t0 = time.time()
        voxel = events_to_voxel_grid(window_events, t_start, t_end)
        t_voxel = time.time() - t0
        t_voxel_total += t_voxel

        voxels.append(voxel)
        x_norm = labels[i, 0] / (W - 1)
        y_norm = labels[i, 1] / (H - 1)
        blink = float(labels[i, 2])
        labels_out.append([x_norm, y_norm, blink])

    if len(voxels) == 0:
        return 0

    t0 = time.time()
    out_voxels = OUTPUT_DIR / split / f"{folder.name}_voxels.npy"
    out_labels = OUTPUT_DIR / split / f"{folder.name}_labels.npy"
    out_voxels.parent.mkdir(exist_ok=True)
    np.save(out_voxels, np.stack(voxels))
    np.save(out_labels, np.array(labels_out))
    t_save = time.time() - t0

    print(f" → {len(voxels)} samples")
    print(f"     load: {t_load:.1f}s | labels: {t_labels:.3f}s | mask: {t_mask*len(labels):.1f}s | voxel: {t_voxel_total:.1f}s | save: {t_save:.1f}s")

    return len(voxels)
# ========================================
# 4. MAIN
# ========================================
if __name__ == "__main__":
    print("STARTING MEMORY-SAFE PREPROCESSING")
    for split in ['train', 'test']:
        split_path = DATA_ROOT / split
        if not split_path.exists():
            print(f"ERROR: {split_path} not found!")
            continue

        folders = sorted([f for f in split_path.iterdir() if f.is_dir()])
        print(f"\nPROCESSING {split.upper()} — {len(folders)} recordings")

        total_samples = 0
        for folder in tqdm(folders, desc=split):
            n = preprocess_recording(folder, split)
            total_samples += n

        print(f"{split.upper()} DONE: {total_samples} samples saved in {OUTPUT_DIR}/{split}/")

    print("\nPREPROCESSING COMPLETE!")
    print("Next: python 2_train.py (will merge on-the-fly)")