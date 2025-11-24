# 1_preprocess.py — COMPACT: ONE .pt + ONE .txt PER RECORDING
import h5py
import numpy as np
import torch
from pathlib import Path
import re
from tqdm import tqdm

# ========================================
# CONFIG
# ========================================
T_LABELED = 10000
H, W = 480, 640

# Jetson paths
# DATA_ROOT = Path("/home/jetson/Jon/IndustrialProject/akida_examples/1_ec_example/training/event-based-eye-tracking-cvpr-2025/3ET+ dataset/event_data")
# OUTPUT_ROOT = Path("/home/jetson/Jon/IndustrialProject/akida_examples/1_ec_example/training/preprocessed")

# Alienware paths
DATA_ROOT = Path("/home/dronelab-pc-1/Jon/IndustrialProject/akida_examples/1_ec_example/training/event-based-eye-tracking-cvpr-2025/3ET+ dataset/event_data")
OUTPUT_ROOT = Path("/home/dronelab-pc-1/Jon/IndustrialProject/akida_examples/1_ec_example/training/preprocessed")
OUTPUT_ROOT.mkdir(exist_ok=True)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
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
# 2. GPU VOXEL
# ========================================
def events_to_voxel_gpu(events, t_start, t_end):
    if len(events) == 0:
        return torch.zeros((W, H), device=DEVICE)

    t = torch.from_numpy(events['t'].astype(np.float64)).to(DEVICE)
    x = torch.from_numpy(events['x']).to(DEVICE, dtype=torch.long)
    y = torch.from_numpy(events['y']).to(DEVICE, dtype=torch.long)
    p = torch.from_numpy(events['p']).to(DEVICE, dtype=torch.long)

    mask = (t >= t_start) & (t <= t_end)
    if not mask.any():
        return torch.zeros((W, H), device=DEVICE)

    t, x, y, p = t[mask], x[mask], y[mask], p[mask]

    valid = (x >= 0) & (x < W) & (y >= 0) & (y < H)
    t, x, y, p = t[valid], x[valid], y[valid], p[valid]
    if len(t) == 0:
        return torch.zeros((W, H), device=DEVICE)

    # Positive events are p=0 -> +1 ; negative events are p=1 -> -1
    # pol = torch.where(p == 1, -1.0, 1.0)
    pol = torch.where(p == 1, torch.tensor(-1, dtype=torch.int8, device=DEVICE),
                         torch.tensor( 1, dtype=torch.int8, device=DEVICE))

    # Flatten spatial index: y * W + x -> This works for flat_voxel.view(H, W)
    # Flatten spatial index: x * H + y -> This works for flat_voxel.view(W, H) -> USE THIS
    flat_idx = x * H + y

    # flat_voxel = torch.zeros(1, H * W, device=DEVICE, dtype=torch.float32)
    flat_voxel = torch.zeros(1, H * W, device=DEVICE, dtype=torch.int8)
    flat_idx = flat_idx.unsqueeze(0)
    pol = pol.unsqueeze(0)
    flat_voxel.scatter_add_(1, flat_idx, pol)

    return flat_voxel.view(W, H)

# ========================================
# 3. PREPROCESS ONE RECORDING → ONE FILE
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

    sort_idx = np.argsort(events['t'])
    events = events[sort_idx]
    t_events = events['t']

    t_labels_us = (np.arange(len(labels)) + 1) * T_LABELED

    # Collect all voxels
    voxel_list = []
    valid_labels = []

    for i in range(len(labels)):
        t_label = t_labels_us[i]
        t_start = max(0, t_label - T_LABELED)
        t_end = t_label

        left = np.searchsorted(t_events, t_start, side='left')
        right = np.searchsorted(t_events, t_end, side='right')
        if left >= right:
            continue

        window_events = events[left:right]
        voxel = events_to_voxel_gpu(window_events, t_start, t_end)
        voxel_list.append(voxel.cpu())  # collect on CPU
        valid_labels.append([t_label, labels[i][0], labels[i][1], labels[i][2]])

    if len(voxel_list) == 0:
        return 0

    # Stack all voxels: [N, 640, 480]
    voxel_stack = torch.stack(voxel_list)  # [N, W, H]

    # Save
    pt_path = out_dir / "voxels.pt"
    txt_path = out_dir / "labels.txt"

    torch.save(voxel_stack, pt_path)

    with open(txt_path, 'w') as f:
        for time, x, y, state in valid_labels:
            f.write(f"{time} {x} {y} {state}\n")

    return len(voxel_list)

# ========================================
# 4. MAIN
# ========================================
if __name__ == "__main__":
    print("STARTING COMPACT PREPROCESSING (1 .pt + 1 .txt per recording)")
    print(f"Input: {DATA_ROOT}")
    print(f"Output: {OUTPUT_ROOT}")
    print(f"Format: voxels.pt [N,640,480] + labels.txt [N×4]")

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
                tqdm.write(f" {folder.name}: {n} samples → voxels.pt")
    print(f"\nPREPROCESSING COMPLETE! Total samples: {total_samples}")
    print("Files created:")
    print(" preprocessed/train/REC_ID/voxels.pt")
    print(" preprocessed/train/REC_ID/labels.txt")