# _1_4_preprocess_akida.py
# Exact replication of BrainChip Akida gaze example preprocessing
# → Input:  [N, 2, 50, 96, 128]   (pos/neg binary, 500ms)
# → Output: [N, 3, 50, 3, 4]       (target heatmap: confidence + x/y offset)
# → Only keeps open-eye samples (state == 0)

import h5py
import numpy as np
import torch
from pathlib import Path
import re
from tqdm import tqdm

# ========================================
# CONFIG — Akida-exact parameters
# ========================================
T_LABELED = 10000                  # 10 ms per bin
N_BINS = 50                    # 50 bins = 500 ms
H_ORIG, W_ORIG = 480, 640
H_DOWN, W_DOWN = 96, 128       # Downsampled resolution (5× smaller)
H_OUT, W_OUT = 3, 4            # Final feature map size after 32× downsampling

# DATA_ROOT = Path("/home/dronelab-pc-1/Jon/IndustrialProject/akida_examples/1_ec_example/training/event-based-eye-tracking-cvpr-2025/3ET+ dataset/event_data")
# OUTPUT_ROOT = Path("/home/dronelab-pc-1/Jon/IndustrialProject/akida_examples/1_ec_example/training/preprocessed_akida")
DATA_ROOT = Path("/home/pi/Jon/IndustrialProject/akida_examples/1_ec_example/training/event-based-eye-tracking-cvpr-2025/3ET+ dataset/event_data")
OUTPUT_ROOT = Path("/home/pi/Jon/IndustrialProject/akida_examples/1_ec_example/training/preprocessed_akida")
OUTPUT_ROOT.mkdir(exist_ok=True)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Preprocessing for Akida-exact model (T=50, 128×96, target heatmaps)")

# ========================================
# 1. Load labels (x, y, state) at 100 Hz
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
    return np.array(labels, dtype=np.int32)  # [N, 3]

# ========================================
# 2. Fast GPU voxel grid (single bin)
# ========================================
def events_to_voxel_gpu(events, t_start, t_end):
    if len(events) == 0:
        return torch.zeros((2, H_DOWN, W_DOWN), device=DEVICE, dtype=torch.uint8)

    t = torch.from_numpy(events['t'].astype(np.float64)).to(DEVICE)
    x = torch.from_numpy(events['x']).to(DEVICE, dtype=torch.long)
    y = torch.from_numpy(events['y']).to(DEVICE, dtype=torch.long)
    p = torch.from_numpy(events['p']).to(DEVICE, dtype=torch.long)

    mask = (t >= t_start) & (t < t_end)
    if not mask.any():
        return torch.zeros((2, H_DOWN, W_DOWN), device=DEVICE, dtype=torch.uint8)

    x, y, p = x[mask], y[mask], p[mask]

    # Downsample coordinates: 640×480 → 128×96
    x_down = (x.float() * (W_DOWN / W_ORIG)).long()
    y_down = (y.float() * (H_DOWN / H_ORIG)).long()

    valid = (x_down >= 0) & (x_down < W_DOWN) & (y_down >= 0) & (y_down < H_DOWN)
    x_down, y_down, p = x_down[valid], y_down[valid], p[valid]

    if len(x_down) == 0:
        return torch.zeros((2, H_DOWN, W_DOWN), device=DEVICE, dtype=torch.uint8)

    # Two channels: pos (p==0), neg (p==1)
    voxel = torch.zeros((2, H_DOWN, W_DOWN), device=DEVICE, dtype=torch.uint8)
    pos_mask = p == 0
    neg_mask = p == 1

    # Flatten spatial index: y * W + x -> This works for flat_voxel.view(H, W) -> USE THIS
    # Flatten spatial index: x * H + y -> This works for flat_voxel.view(W, H) 
    if pos_mask.any():
        idx = y_down[pos_mask] * W_DOWN + x_down[pos_mask]
        # positive in channel 0
        voxel.view(2, -1)[0].index_add_(0, idx, torch.ones_like(idx, dtype=torch.uint8))
    if neg_mask.any():
        idx = y_down[neg_mask] * W_DOWN + x_down[neg_mask]
        # negative in channel 1
        voxel.view(2, -1)[1].index_add_(0, idx, torch.ones_like(idx, dtype=torch.uint8))

    # [2, 96, 128] uint8
    return voxel

# ========================================
# 3. Create heatmap
# ========================================

def create_heatmap(x_out, y_out):
    """
    x_out: float in [0, 4)   → position in final W=4 grid
    y_out: float in [0, 3)   → position in final H=3 grid
    
    Returns: torch.Tensor [3, 3, 4] → (confidence, y_offset, x_offset)
    """
    heatmap = torch.zeros(3, 3, 4)  # [C, H, W]

    # Fractional part = relative offset inside the cell
    x_idx = int(x_out)        # integer part → column
    y_idx = int(y_out)        # integer part → row
    x_offset = x_out - x_idx  # fractional → x_offset ∈ [0,1)
    y_offset = y_out - y_idx  # fractional → y_offset ∈ [0,1)

    # Only activate the cell where the gaze falls
    if 0 <= y_idx < 3 and 0 <= x_idx < 4:
        heatmap[0, y_idx, x_idx] = 1.0          # confidence = 1 at the correct cell
        heatmap[1, y_idx, x_idx] = y_offset     # y relative offset
        heatmap[2, y_idx, x_idx] = x_offset     # x relative offset

    return heatmap  # [3, 3, 4]

# ========================================
# 4. Preprocess one recording
# ========================================
def preprocess_recording(folder, split, save_root):
    rec_id = folder.name
    h5_file = folder / f"{rec_id}.h5"
    txt_file = folder / "label.txt"

    if not h5_file.exists() or not txt_file.exists():
        return 0

    out_dir = save_root / split / rec_id
    out_dir.mkdir(parents=True, exist_ok=True)

    with h5py.File(h5_file, 'r') as f:
        events = f['events'][()]

    labels = load_labels(txt_file)
    if len(labels) < N_BINS:
        return 0

    # Sort events once
    sort_idx = np.argsort(events['t'])
    events = events[sort_idx]
    t_events = events['t']

    voxel_list = []
    heatmap_list = []
    valid_count = 0

    for i in range(N_BINS - 1, len(labels)):
        # closed eye → skip
        if labels[i][2] == 1:  
            continue

        t_end = (i + 1) * T_LABELED  # 100 Hz labels → 10ms steps
        t_start = t_end - N_BINS * T_LABELED

        # Build 50-bin voxel + 50 individual heatmaps (one per bin)
        voxel_bins = []
        heatmap_bins = []
        skip = False

        for b in range(N_BINS):
            b_start = t_start + b * T_LABELED
            b_end = b_start + T_LABELED

            left = np.searchsorted(t_events, b_start, side='left')
            right = np.searchsorted(t_events, b_end, side='right')
            bin_events = events[left:right] if left < right else events[0:0]

            voxel_bin = events_to_voxel_gpu(bin_events, b_start, b_end)
            voxel_bins.append(voxel_bin.cpu())

            # label index = first label (0) + (i - 49) + b
            label_idx = (i - (N_BINS - 1)) + b
            if labels[label_idx][2] == 1:  # closed eye → skip whole sample
                skip = True
                break 

            x_px, y_px = labels[label_idx][0], labels[label_idx][1]
            x_out = x_px * (W_OUT / W_ORIG)
            y_out = y_px * (H_OUT / H_ORIG)

            heatmap_bin = create_heatmap(x_out, y_out)  # [3, 3, 4]
            heatmap_bins.append(heatmap_bin)
        
        if skip:
            continue

        # Only add if all 50 bins were valid (open eye)
        voxel_tensor = torch.stack(voxel_bins)  # [50, 2, 96, 128] → transpose
        voxel_tensor = voxel_tensor.permute(1, 0, 2, 3)  # [2, 50, 96, 128]

        heatmap_tensor = torch.stack(heatmap_bins)  # [50, 3, 3, 4] → transpose
        heatmap_tensor = heatmap_tensor.permute(1, 0, 2, 3)   # [3, 50, 3, 4]

        voxel_list.append(voxel_tensor)
        heatmap_list.append(heatmap_tensor)
        valid_count += 1

    if valid_count == 0:
        return 0

    # Final tensors
    voxels_final = torch.stack(voxel_list)      # [N, 2, 50, 96, 128]
    heatmaps_final = torch.stack(heatmap_list)  # [N, 3, 50, 3, 4]

    # Save
    torch.save(voxels_final.to(torch.uint8), out_dir / "voxels.pt")
    torch.save(heatmaps_final, out_dir / "heatmaps.pt")

    return valid_count

# ========================================
# 5. MAIN
# ========================================
if __name__ == "__main__":
    print("Starting Akida-exact preprocessing (T=50, 128×96, target heatmaps)")
    total = 0
    for split in ['train', 'test']:
        split_path = DATA_ROOT / split
        if not split_path.exists():
            print(f"Split {split} not found!")
            continue

        folders = sorted([f for f in split_path.iterdir() if f.is_dir()])
        print(f"\nProcessing {split.upper()} — {len(folders)} recordings")

        for folder in tqdm(folders):
            n = preprocess_recording(folder, split, OUTPUT_ROOT)
            if n > 0:
                total += n
                tqdm.write(f"  {folder.name}: {n} samples → voxels.pt + heatmaps.pt")

    print(f"\nDONE! Total open-eye samples: {total}")
    print(f"Output folder: {OUTPUT_ROOT}")