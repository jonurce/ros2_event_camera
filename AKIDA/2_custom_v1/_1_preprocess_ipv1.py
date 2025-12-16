# 1_preprocess_v1.py — FOR v1-COMPATIBLE MODEL (NO TEMPORAL DIMENSION)
# Changes for v1 model:
# - Remove T=10 bins → use single accumulation window per label
# - Keep polarity: Channel 0 = positive events, Channel 1 = negative events
# - Downsample to 96x128
# - Normalize to [0,1] float32
# - Output: voxels.pt [N, 2, 96, 128], labels.txt [N, 4]

import h5py
import numpy as np
import torch
from pathlib import Path
import re
from tqdm import tqdm
from torchvision import transforms

# ========================================
# CONFIG
# ========================================
T_LABELED = 10000  # 10 ms per label
H_ORIG, W_ORIG = 480, 640
H_TARGET, W_TARGET = 96, 128  # Model input size

# Paths (update as needed)
DATA_ROOT = Path("/home/dronelab-pc-1/Jon/IndustrialProject/AKIDA/1_ec_example/training/event-based-eye-tracking-cvpr-2025/3ET+ dataset/event_data")
OUTPUT_ROOT = Path("/home/dronelab-pc-1/Jon/IndustrialProject/AKIDA/2_custom_v1/preprocessed_v1")
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
# 2. EVENT TO VOXEL GRID (2 channels: pos/neg, normalized [0,1])
# ========================================
def events_to_voxel_grid(events, t_start, t_end):
    if len(events) == 0:
        return torch.zeros((2, H_TARGET, W_TARGET), device=DEVICE, dtype=torch.float32)

    t = torch.from_numpy(events['t'].astype(np.float64)).to(DEVICE)
    x = torch.from_numpy(events['x']).to(DEVICE, dtype=torch.long)
    y = torch.from_numpy(events['y']).to(DEVICE, dtype=torch.long)
    p = torch.from_numpy(events['p']).to(DEVICE, dtype=torch.long)

    mask = (t >= t_start) & (t < t_end)
    if not mask.any():
        return torch.zeros((2, H_TARGET, W_TARGET), device=DEVICE, dtype=torch.float32)

    x, y, p = x[mask], y[mask], p[mask]

    # Downsample coordinates: 640×480 → 128×96
    x_down = (x.float() * (W_TARGET / W_ORIG)).long()
    y_down = (y.float() * (H_TARGET / H_ORIG)).long()

    valid = (x_down >= 0) & (x_down < W_TARGET) & (y_down >= 0) & (y_down < H_TARGET)
    x_down, y_down, p = x_down[valid], y_down[valid], p[valid]

    if len(x_down) == 0:
        return torch.zeros((2, H_TARGET, W_TARGET), device=DEVICE, dtype=torch.float32)

    # Two channels: pos (p==0), neg (p==1)
    voxel = torch.zeros((2, H_TARGET, W_TARGET), device=DEVICE, dtype=torch.float32)
    pos_mask = p == 0
    neg_mask = p == 1

    # Flatten spatial index: y * W + x -> This works for flat_voxel.view(H, W) -> USE THIS
    # Flatten spatial index: x * H + y -> This works for flat_voxel.view(W, H) 
    if pos_mask.any():
        idx = y_down[pos_mask] * W_TARGET + x_down[pos_mask]
        # positive in channel 0
        voxel.view(2, -1)[0].index_add_(0, idx, torch.ones_like(idx, dtype=torch.float32))
    if neg_mask.any():
        idx = y_down[neg_mask] * W_TARGET + x_down[neg_mask]
        # negative in channel 1
        voxel.view(2, -1)[1].index_add_(0, idx, torch.ones_like(idx, dtype=torch.float32))

    # voxel: [2, 96, 128] float32

    # Normalize to [0,1] using the GLOBAL maximum across both channels
    global_max = voxel.max()  # Single scalar: highest value in the entire [2, H, W] voxel
    global_max = torch.clamp(global_max, min=1.0)  # Avoid division by zero
    voxel = voxel / global_max  # Now all values in [0, 1]
    voxel = torch.clamp(voxel, min=0.0, max=1.0)

    return voxel


# ========================================
# 3. Create heatmap
# ========================================

def create_heatmap(x_norm, y_norm):
    """
    x_norm: float in [0, 4)   → position normalized
    y_norm: float in [0, 3)   → position normalized
    
    Returns: torch.Tensor [3] → (confidence, y_offset, x_offset)
    """
    heatmap = torch.zeros(3, 1, 1)  # [C, H, W]

    heatmap[0] = 1.0        # confidence = 1 (eye in frame)
    heatmap[1] = y_norm     # y relative offset
    heatmap[2] = x_norm     # x relative offset

    return heatmap  # [3]

# ========================================
# 3. PREPROCESS ONE RECORDING
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

    # Sort events by time
    sort_idx = np.argsort(events['t'])
    events = events[sort_idx]
    t_events = events['t']

    # Convert label times to μs
    t_labels_us = (np.arange(len(labels)) + 1) * T_LABELED

    voxel_list = []
    heatmap_list = []

    for i in range(len(labels)):
        # Skip closed eyes (blinks)
        if labels[i][2] == 1:
            continue

        t_label = t_labels_us[i]
        t_start = max(0, t_label - T_LABELED)
        t_end = t_label

        left = np.searchsorted(t_events, t_start, side='left')
        right = np.searchsorted(t_events, t_end, side='right')
        if left >= right:
            continue

        window_events = events[left:right]
        voxel = events_to_voxel_grid(window_events, t_start, t_end)
        voxel_list.append(voxel.cpu())

        x_norm = np.clip(labels[i][0] / W_ORIG, 0.0, 1.0)
        y_norm = np.clip(labels[i][1] / H_ORIG, 0.0, 1.0)
        heatmap_bin = create_heatmap(x_norm, y_norm)  # [3]
        heatmap_list.append(heatmap_bin)

    if len(voxel_list) == 0:
        return 0

    # Stack: [N, 2, 96, 128]
    voxel_stack = torch.stack(voxel_list)

    # Stack heatmaps: [N, 3]
    heatmap_tensor = torch.stack(heatmap_list)

    # Save
    torch.save(voxel_stack, out_dir / "voxels.pt")
    torch.save(heatmap_tensor, out_dir / "heatmaps.pt")

    return len(voxel_list)

# ========================================
# 4. MAIN
# ========================================
if __name__ == "__main__":
    print("STARTING PREPROCESSING FOR v1-COMPATIBLE MODEL (NO TEMPORAL DIMENSION)")
    print(f"Input: {DATA_ROOT}")
    print(f"Output: {OUTPUT_ROOT}")
    print(f"Format: voxels.pt [N, 2, 96, 128] float32 normalized + labels.txt")

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

    print(f"\nPREPROCESSING COMPLETE! Total samples: {total_samples}")