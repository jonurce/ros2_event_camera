# 1_preprocess.py — GPU-ACCELERATED, .pt + .txt OUTPUT
# → For each label: save voxel.pt (10,640,480) + label.txt (x y state)
# → Uses last 100ms events before label
# → Saves to: preprocessed/train/REC_ID/sample_XXX.pt + .label.txt

import h5py
import numpy as np
import torch
from pathlib import Path
import re
from tqdm import tqdm
import time

# ========================================
# CONFIG
# ========================================
T_BINS = 10
T_LABELED = 10000 # Labeled at 100Hz = 0.01s = 10000us
WINDOW_US = T_BINS * T_LABELED 
H, W = 480, 640
DATA_ROOT = Path("/home/jetson/Jon/IndustrialProject/akida_examples/1_ec_example/training/event-based-eye-tracking-cvpr-2025/3ET+ dataset/event_data")
OUTPUT_ROOT = Path("/home/jetson/Jon/IndustrialProject/akida_examples/1_ec_example/training/preprocessed")
OUTPUT_ROOT.mkdir(exist_ok=True)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# DEVICE = "cpu"
print(f"Using device: {DEVICE}")

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
# 2. GPU-ACCELERATED VOXEL GRID (TORCH)
# ========================================
def events_to_voxel_gpu(events, t_start, t_end):
    """
    Input: events (N,) with .t, .x, .y, .p
    Output: torch.Tensor (10, 640, 480) on GPU → +1 / -1
    """
    t = torch.from_numpy(events['t'].astype(np.float64)).to(DEVICE)
    x = torch.from_numpy(events['x']).to(DEVICE, dtype=torch.long)
    y = torch.from_numpy(events['y']).to(DEVICE, dtype=torch.long)
    p = torch.from_numpy(events['p']).to(DEVICE, dtype=torch.long)

    # Clamp time to [t_start, t_end] - Double check, as input already comes clamped
    mask = (t >= t_start) & (t <= t_end)
    if not mask.any():
        return torch.zeros((T_BINS, W, H), dtype=torch.float32)

    t, x, y, p = t[mask], x[mask], y[mask], p[mask]

    # Validate coordinates
    valid = (x >= 0) & (x < W) & (y >= 0) & (y < H)
    if not valid.all():
        t, x, y, p = t[valid], x[valid], y[valid], p[valid]

    if len(t) == 0:
        return torch.zeros((T_BINS, W, H), dtype=torch.float32)

    # Time → bin index
    t_norm = T_BINS * (t - t_start) / (t_end - t_start + 1e-6)
    t_norm = torch.clamp(t_norm, 0, T_BINS - 1)
    bin_idx = t_norm.floor().long() # (N,)

    # Polarity
    pol = torch.where(p == 1, 1.0, -1.0).to(DEVICE) # (N,)

    # Flatten spatial index: bin_idx * (H*W) + y * W + x -> This works for flat_voxel.view(T_BINS, H, W)
    # Flatten spatial index: bin_idx * (H*W) + x * H + y -> This works for flat_voxel.view(T_BINS, W, H) -> USE THIS
    flat_idx = bin_idx * (H * W) + x * H + y # (N,)

    # Create flat voxel: (T_BINS, H*W)
    flat_voxel = torch.zeros(1, T_BINS * H * W, device=DEVICE, dtype=torch.float32)

    flat_idx = flat_idx.unsqueeze(0)  # (1, N)
    pol = pol.unsqueeze(0)            # (1, N)

    # Scatter: src must be same size as index
    flat_voxel.scatter_add_(1, flat_idx, pol)

    return flat_voxel.view(T_BINS, W, H)#.cpu() # (10,640,480)

# ========================================
# 3. PREPROCESS ONE RECORDING
# ========================================
def preprocess_recording(folder, split):
    rec_id = folder.name
    h5_file = folder / f"{rec_id}.h5"
    txt_file = folder / "label.txt"

    if not h5_file.exists() or not txt_file.exists():
        return 0

    # Output dir
    out_dir = OUTPUT_ROOT / split / rec_id
    out_dir.mkdir(parents=True, exist_ok=True)

    # Load data
    with h5py.File(h5_file, 'r') as f:
        events = f['events'][()]

    labels = load_labels(txt_file)
    if len(labels) == 0:
        return 0

    # Sort events by time
    sort_idx = np.argsort(events['t'])
    events = events[sort_idx]
    t_events = events['t']

    # Label times (100Hz)
    t_labels_us = (np.arange(len(labels)) + 1) * T_LABELED

    sample_count = 0
    for i in range(len(labels)):
        t_label = t_labels_us[i]
        t_start = max(0, t_label - WINDOW_US)
        t_end = t_label

        # Binary search (indexes of limit times)
        left = np.searchsorted(t_events, t_start, side='left')
        right = np.searchsorted(t_events, t_end, side='right')
        if left >= right:
            continue

        window_events = events[left:right]

        # GPU voxel
        voxel_tensor = events_to_voxel_gpu(window_events, t_start, t_end)  # (10,640,480)

        # Save .pt (tensor) + .label.txt
        sample_name = f"sample_{sample_count:06d}"
        pt_path = out_dir / f"{sample_name}.pt"
        txt_path = out_dir / f"{sample_name}.label.txt"

        torch.save(voxel_tensor, pt_path)

        with open(txt_path, 'w') as f:
            x, y, state = labels[i]
            f.write(f"{x} {y} {state}\n")

        sample_count += 1

    return sample_count

# ========================================
# 4. MAIN
# ========================================
if __name__ == "__main__":
    print("STARTING GPU-ACCELERATED PREPROCESSING")
    print(f"Input:  {DATA_ROOT}")
    print(f"Output: {OUTPUT_ROOT}")
    print(f"Window: {WINDOW_US:.0f}us → {T_BINS} bins")
    print(f"Format: .pt ({T_BINS},{W},{H}) + .label.txt")

    total_samples = 0
    for split in ['train', 'test']:
        split_path = DATA_ROOT / split
        if not split_path.exists():
            print(f"ERROR: {split_path} not found!")
            continue

        folders = sorted([f for f in split_path.iterdir() if f.is_dir()])
        print(f"\nPROCESSING {split.upper()} — {len(folders)} recordings")

        for folder in tqdm(folders, desc=split):
            n = preprocess_recording(folder, split)
            total_samples += n
            if n > 0:
                tqdm.write(f"  {folder.name}: {n} samples")

        print(f"{split.upper()} DONE: {total_samples} total samples")

    print("\nPREPROCESSING COMPLETE!")
    print("Files created:")
    print(" preprocessed/SPLIT/REC_ID/sample_XXX.pt")
    print(" preprocessed/SPLIT/REC_ID/sample_XXX.label.txt")
    print("\nNext: python 2_train.py (on-the-fly from .pt + .txt)")