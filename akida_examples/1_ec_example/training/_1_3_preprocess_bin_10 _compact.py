# _1_3_preprocess_bin_10_compact.py
# ULTRA-COMPACT VERSION: [N, 640, 480] float16
# → Each sample = sum of events over the last 100 ms (10 bins) → only ONE 2D slice per label
# → 10× smaller on disk → 3–5× faster training → same performance (or very close)

import h5py
import numpy as np
import torch
from pathlib import Path
import re
from tqdm import tqdm

# ========================================
# CONFIG
# ========================================
T_BIN = 10000           # 10 ms per bin
N_BINS = 10             # we want 10 consecutive bins per sample
H, W = 480, 640

# Jetson paths
# DATA_ROOT = Path("/home/jetson/Jon/IndustrialProject/akida_examples/1_ec_example/training/event-based-eye-tracking-cvpr-2025/3ET+ dataset/event_data")
# OUTPUT_ROOT = Path("/home/jetson/Jon/IndustrialProject/akida_examples/1_ec_example/training/preprocessed")

# Alienware paths
DATA_ROOT = Path("/home/dronelab-pc-1/Jon/IndustrialProject/akida_examples/1_ec_example/training/event-based-eye-tracking-cvpr-2025/3ET+ dataset/event_data")
OUTPUT_ROOT = Path("/home/dronelab-pc-1/Jon/IndustrialProject/akida_examples/1_ec_example/training/preprocessed_fast_compact")
OUTPUT_ROOT.mkdir(exist_ok=True)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {DEVICE}")

# ========================================
# 1. LABEL LOADER (unchanged)
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
# 2. GPU VOXEL — returns float16 
# ========================================
def events_to_voxel_gpu(events, t_start, t_end):
    if len(events) == 0:
        return torch.zeros((W, H), device=DEVICE, dtype=torch.float16)
    
    t = torch.from_numpy(events['t'].astype(np.float64)).to(DEVICE)
    x = torch.from_numpy(events['x']).to(DEVICE, dtype=torch.long)
    y = torch.from_numpy(events['y']).to(DEVICE, dtype=torch.long)
    p = torch.from_numpy(events['p']).to(DEVICE, dtype=torch.long)

    mask = (t >= t_start) & (t <= t_end)
    if not mask.any():
        return torch.zeros((W, H), device=DEVICE, dtype=torch.float16)

    t, x, y, p = t[mask], x[mask], y[mask], p[mask]

    valid = (x >= 0) & (x < W) & (y >= 0) & (y < H)
    t, x, y, p = t[valid], x[valid], y[valid], p[valid]
    if len(t) == 0:
        return torch.zeros((W, H), device=DEVICE, dtype=torch.float16)

    # Positive events are p=0 -> +1 ; negative events are p=1 -> -1; → directly in float16
    pol = torch.where(p == 1, torch.tensor(-1.0, dtype=torch.float16, device=DEVICE),
                            torch.tensor( 1.0, dtype=torch.float16, device=DEVICE))

    # Flatten spatial index: y * W + x -> This works for flat_voxel.view(H, W)
    # Flatten spatial index: x * H + y -> This works for flat_voxel.view(W, H) -> USE THIS
    flat_idx = x * H + y

    flat_voxel = torch.zeros(1, H * W, device=DEVICE, dtype=torch.float16)
    flat_idx = flat_idx.unsqueeze(0)
    pol = pol.unsqueeze(0)
    flat_voxel.scatter_add_(1, flat_idx, pol)

    return flat_voxel.view(W, H)

# ========================================
# 3. PREPROCESS ONE RECORDING → [N, 640, 480] accumulated 100ms voxel
# ========================================
def preprocess_recording(folder, split):
    rec_id = folder.name
    h5_file = folder / f"{rec_id}.h5"
    txt_file = folder / "label.txt"

    if not h5_file.exists() or not txt_file.exists():
        return 0

    out_dir = OUTPUT_ROOT / split / rec_id
    out_dir.mkdir(parents=True, exist_ok=True)

    with h5py.File(h5_file, 'r') as f:
        events = f['events'][()]

    labels = load_labels(txt_file)
    if len(labels) == 0:
        return 0

    # Sort events once
    sort_idx = np.argsort(events['t'])
    events = events[sort_idx]
    t_events = events['t']

    accumulated_voxels = []      # list of [640, 480] float16
    valid_labels_out = []

    for i in range(N_BINS - 1, len(labels)):  # start when we have 10 bins
        t_label = (i + 1) * T_BIN
        t_start = t_label - N_BINS * T_BIN   # first bin start
        t_end   = t_label                    # last bin end

        # Find events in the full 100 ms (N_BINS * T_BINS) window
        left  = np.searchsorted(t_events, t_start, side='left')
        right = np.searchsorted(t_events, t_end,   side='right')
        if left >= right:
            continue

        window_events = events[left:right]

        # → ONE SINGLE VOXEL: accumulate all events in 100ms
        voxel = events_to_voxel_gpu(window_events, t_start, t_end)  # [640, 480]
        accumulated_voxels.append(voxel.cpu())
        valid_labels_out.append([t_label, labels[i][0], labels[i][1], labels[i][2]])


    if len(accumulated_voxels) == 0:
        return 0

    # Final tensor: [N, 640, 480] float16  ← W=640, H=480
    voxel_tensor = torch.stack(accumulated_voxels) #[N, W, H]
    voxel_tensor = voxel_tensor.contiguous()  # clean memory layout

    pt_path = out_dir / "voxels.pt"
    txt_path = out_dir / "labels.txt"

    torch.save(voxel_tensor, pt_path)  # ← float16
    with open(txt_path, 'w') as f:
        for t, x, y, state in valid_labels_out:
            f.write(f"{t} {x} {y} {state}\n")

    return len(accumulated_voxels)

# ========================================
# 4. MAIN
# ========================================
if __name__ == "__main__":
    print("STARTING ULTRA-COMPACT PREPROCESSING → [N,640,480] float16")
    print("→ Each sample = accumulated events over last 100 ms (10 bins)")
    print("→ ~10× smaller → 3–5× faster training!")
    print(f"Input:  {DATA_ROOT}")
    print(f"Output: {OUTPUT_ROOT}")

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
                tqdm.write(f" {folder.name}: {n} samples → voxels.pt ([N,640,480] float16)")

    print(f"\nPREPROCESSING COMPLETE! Total samples: {total_samples}")
    print(f"Dataset ready: {OUTPUT_ROOT}")