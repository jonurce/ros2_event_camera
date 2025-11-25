# _4_1_test_tennst_plot_akida.py
# Visualize one sample from the NEW Akida-style dataset
# → Shows: 2D/3D events + soft-argmax prediction vs ground truth
# → Uses uint8 voxel input [2,50,96,128] + heatmap target [3,50,3,4]

import torch
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from pathlib import Path
from tqdm import tqdm

# ←←← YOUR NEW MODEL AND DATASET ←←←
from _2_5_model_uint8_akida import EyeTennSt
from _3_4_train_fast_akida import AkidaGazeDataset, extract_gaze

# ============================================================
# CONFIG — CHANGE THESE
# ============================================================
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# PATHS
DATA_ROOT = Path("/home/dronelab-pc-1/Jon/IndustrialProject/akida_examples/1_ec_example/training/preprocessed_akida")
MODEL_PATH = Path("/home/dronelab-pc-1/Jon/IndustrialProject/akida_examples/1_ec_example/training/runs/tennst_11_akida_b128_e100/best.pth")

# Which sample to visualize
SPLIT = "test"
REC_ID = "1_1"           # folder name inside test/
SAMPLE_IDX = 200         # index inside that recording

W, H = 640, 480         # original image size
W_IN, H_IN = 128, 96    # Akida input size
W_OUT, H_OUT = 4, 3     # final feature map size
PIXEL_SCALE = (W + H) / (W_OUT + H_OUT)
T = 50  # number of time bins

print(f"Visualizing: {SPLIT}/{REC_ID} — sample {SAMPLE_IDX}")

# ============================================================
# Load model
# ============================================================
model = EyeTennSt(t_kernel_size=5, s_kernel_size=3, n_depthwise_layers=4).to(DEVICE)

if torch.cuda.device_count() > 1:
    print(f"Using {torch.cuda.device_count()} GPUs with DataParallel")
    model = torch.nn.DataParallel(model)

print("Compiling model...")
model = torch.compile(model, mode="reduce-overhead")

# Load weights (handles checkpoint dict)
checkpoint = torch.load(MODEL_PATH, map_location=DEVICE)

if 'model_state_dict' in checkpoint:
    state_dict = checkpoint['model_state_dict']
    print(f"Loaded checkpoint from epoch {checkpoint.get('epoch', '?')}")
    print(f"   → val_loss: {checkpoint.get('val_loss', 'N/A'):.4f} | val_loss_l2: {checkpoint.get('val_loss_l2', 'N/A'):.3f}")
else:
    state_dict = checkpoint  # old format: direct state_dict

if isinstance(model, torch.nn.DataParallel):
    model.module.load_state_dict(state_dict)
else:
    model.load_state_dict(state_dict)
model.eval()
print("Model loaded")

# ============================================================
# Load ONE sample from AkidaGazeDataset
# ============================================================
dataset = AkidaGazeDataset(split=SPLIT)

found_sample = None
cumulative_idx = 0

for rec_dir in sorted((DATA_ROOT / SPLIT).iterdir()):
    if not rec_dir.is_dir():
        continue

    heatmaps_path = rec_dir / "heatmaps.pt"
    if not heatmaps_path.exists():
        continue
    
    # Load only the length (fast!)
    heatmaps = torch.load(heatmaps_path, map_location="cpu", mmap=True, weights_only=True)
    N = len(heatmaps)
    
    if rec_dir.name == REC_ID:   # ← compare .name (string), not Path
        if SAMPLE_IDX >= N:
            raise ValueError(f"Recording {REC_ID} has only {N} samples, you asked for {SAMPLE_IDX}")
        
        # Found it! Load the exact sample
        voxels = torch.load(rec_dir / "voxels.pt", map_location="cpu", mmap=True, weights_only=True)[SAMPLE_IDX]
        heatmap = heatmaps[SAMPLE_IDX]
        
        found_sample = {
            'input': voxels,           # [2,50,96,128]
            'target': heatmap,         # [3,50,3,4]
            'rec_id': rec_dir.name,
            'idx_in_rec': SAMPLE_IDX
        }
        break
    
    cumulative_idx += N

if found_sample is None:
    raise ValueError(f"Recording {REC_ID} not found in {SPLIT}/")

x = found_sample['input'].unsqueeze(0).to(DEVICE)      # [1,2,50,96,128] uint8
target = found_sample['target'].unsqueeze(0).to(DEVICE)  # [1,3,50,3,4]

# ============================================================
# CORRECTED: Visualize the RAW winning cell + offset (not soft-argmax)
# ============================================================
def get_original_gaze_point(heatmap_batch):
    """
    heatmap_batch: [1, 3, 50, 3, 4] (from model output)
    Returns: (x_px, y_px) from the cell with highest confidence + its offsets
    """
    last = heatmap_batch[0, :, -1]          # [3, 3, 4] → last time step
    conf = last[0]                          # [3, 4] confidence logits

    # Find cell with highest confidence
    conf_flat = conf.flatten()
    max_idx = conf_flat.argmax().item()
    y_idx, x_idx = np.unravel_index(max_idx, (H_OUT, W_OUT))  # row, col

    # Get offsets from that cell
    y_offset = last[1, y_idx, x_idx].sigmoid().item() if torch.is_tensor(last[1, y_idx, x_idx]) else last[1, y_idx, x_idx]
    x_offset = last[2, y_idx, x_idx].sigmoid().item() if torch.is_tensor(last[2, y_idx, x_idx]) else last[2, y_idx, x_idx]

    # Convert to pixel coordinates
    cell_x = x_idx * (W / W_OUT)   # 640 
    cell_y = y_idx * (H / H_OUT)   # 480

    x_px = cell_x + x_offset * (W / W_OUT)   # ±160 px max offset
    y_px = cell_y + y_offset * (H / H_OUT)

    return x_px, y_px, (x_idx, y_idx), conf_flat[max_idx].item()



# ============================================================
# Inference
# ============================================================
with torch.no_grad(), torch.amp.autocast('cuda'):
    pred = model(x)         # [1,3,50,3,4]

# PREDICTION: raw winning cell + offset
pred_x, pred_y, pred_cell, pred_conf = get_original_gaze_point(pred)
print(f"Pred → Cell: {pred_cell} | Conf: {pred_conf:.3f} | Offset: ({(pred_x):.1f}, {(pred_y):.1f})")

# GROUND TRUTH: from target heatmap (should be clean)
gt_x, gt_y, gt_cell, gt_conf = get_original_gaze_point(target)
print(f"GT → Cell: {gt_cell} | Conf: {gt_conf:.3f} | Offset: ({(gt_x):.1f}, {(gt_y):.1f})")

# Also compute soft-argmax for comparison (optional)
gaze_pred_soft = extract_gaze(pred).cpu().numpy()[0]
gaze_pred_soft_px = gaze_pred_soft * np.array([W, H])

# Error (using raw point — more honest for visualization)
error_px = np.sqrt((pred_x - gt_x)**2 + (pred_y - gt_y)**2)
print(f"Error (raw cell+offset): {error_px:.2f} px")
print(f"Error (soft-argmax): {np.linalg.norm(gaze_pred_soft_px - np.array([gt_x, gt_y])):.2f} px")

# ============================================================
# Prepare events for plotting (accumulate polarity)
# ============================================================
voxel = found_sample['input'].numpy()                  # [2,50,96,128]
pos = voxel[0]                                         # ON events
neg = voxel[1]                                         # OFF events
accum = pos.astype(np.float32) - neg.astype(np.float32)  # net polarity

# Upsample to 640×480 for nice visualization
from scipy.ndimage import zoom
accum_up = zoom(accum.sum(axis=0), (H/H_IN, W/W_IN), order=1)  # [480,640]

# Custom colormap
cmap = plt.cm.bwr
colors = cmap(np.linspace(0, 1, 256))
colors[127:129] = [1, 1, 1, 1]
custom_cmap = ListedColormap(colors)

# ============================================================
# PLOT 1: 2D Event Map + Gaze
# ============================================================
plt.figure(figsize=(15, 7))

plt.subplot(1, 2, 1)
max_val = max(1, abs(accum_up).max())

sc = plt.imshow(accum_up, cmap=custom_cmap, vmin=-max_val, vmax=max_val, origin='lower')
plt.colorbar(sc, label='Net Events (ON − OFF)', shrink=0.8)
plt.title(f"2D Event Accumulation — {REC_ID} — Sample {SAMPLE_IDX}")
plt.xlabel("X (px)")
plt.ylabel("Y (px)")

# Plot gaze
plt.plot(gt_x, gt_y, 'r+', markersize=12, markeredgewidth=3, label='Ground Truth')
plt.plot(pred_x, pred_y, 'g+', markersize=12, markeredgewidth=3, label='Prediction')
plt.legend(fontsize=12)
plt.xlim(0, 640)
plt.ylim(0, 480)

# ============================================================
# PLOT 2: 3D Spatio-Temporal Voxel
# ============================================================
ax = plt.subplot(1, 2, 2, projection='3d')
for t in range(T):
    layer = accum[t]
    yy, xx = np.where(layer != 0)
    if len(xx) == 0: continue
    zz = np.full_like(xx, t)
    vals = layer[yy, xx]
    ax.scatter(zz, xx*W/W_IN, yy*H/H_IN, c=vals, cmap='bwr', s=2, alpha=0.2, vmin=-1, vmax=1)

# Plot gaze at last time step
ax.scatter([49], [gt_x], [gt_y], c='red', s=200, marker='+', linewidth=3, label='GT')
ax.scatter([49], [pred_x], [pred_y], c='green', s=200, marker='+', linewidth=3, label='Pred')

ax.set_xlabel("Time (0–49)")
ax.set_ylabel("X (px ×5)")
ax.set_zlabel("Y (px ×5)")
ax.set_title("3D Event Voxel")
ax.legend()

plt.suptitle(f"Akida-Style EyeTennSt — Error: {error_px:.2f} px", fontsize=16)
plt.tight_layout()
plt.show()