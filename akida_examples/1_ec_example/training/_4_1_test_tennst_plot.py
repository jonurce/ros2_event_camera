# 4_1_test_tennsts_plot.py
# Load a single sample from the test set → run EyeTennSt → plot 2D + 3D events + prediction vs label
# Works with preprocessed_fast/ [N,10,640,480] float16 dataset

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from _2_model_f16 import EyeTennSt

# ============================================================
# CONFIG — CHANGE THESE
# ============================================================
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Paths
DATA_ROOT = Path("/home/dronelab-pc-1/Jon/IndustrialProject/akida_examples/1_ec_example/training/preprocessed_fast")
MODEL_PATH = Path("/home/dronelab-pc-1/Jon/IndustrialProject/akida_examples/1_ec_example/training/runs/tennst_3_f16_batch_24_epochs_130/best.pth")

# Which sample to visualize (you can change this!)
SPLIT = "test"           # or "train"
REC_ID = "1_1"           # recording folder name
SAMPLE_IDX = 500         # index inside that recording (0 to N-1)

print(f"Visualizing: {SPLIT}/{REC_ID} — sample {SAMPLE_IDX}")

# ============================================================
# Minimal dataset — only loads ONE recording (fast & safe)
# ============================================================
class SingleRecordingDataset(Dataset):
    def __init__(self, rec_path):
        self.voxels = torch.load(rec_path / "voxels.pt", map_location="cpu", mmap=True, weights_only=True)
        self.labels = np.loadtxt(rec_path / "labels.txt", dtype=np.int32)
        print(f"Loaded recording: {len(self.voxels)} samples")

    def __len__(self):
        return len(self.voxels)

    def __getitem__(self, idx):
        window = self.voxels[idx]          # [10, 640, 480] float16
        x, y, state = self.labels[idx, 1:4]
        return {
            'input': window,
            'gaze': torch.tensor([x, y], dtype=torch.float32),
            'state': torch.tensor(state, dtype=torch.float32),
            'idx': idx
        }

# ============================================================
# Load model
# ============================================================
model = EyeTennSt(t_kernel_size=5, s_kernel_size=3, n_depthwise_layers=6).to(DEVICE)

# Support multi-GPU if available
if torch.cuda.device_count() > 1:
    print(f"Using {torch.cuda.device_count()} GPUs")
    model = torch.nn.DataParallel(model)

# TORCH.COMPILE — still great, now even more stable
print("Compiling model with torch.compile()...")
model = torch.compile(model, mode="reduce-overhead")

model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
model.eval()
print("Model loaded")

# ============================================================
# Load selected sample
# ============================================================
rec_path = DATA_ROOT / SPLIT / REC_ID
dataset = SingleRecordingDataset(rec_path)
sample = dataset[SAMPLE_IDX]

x = sample['input'].unsqueeze(0).to(DEVICE)        # [1, 10, 640, 480]
gaze_label = sample['gaze'].numpy()
state_label = int(sample['state'].item())

# Inference
with torch.no_grad(), torch.cuda.amp.autocast():
    gaze_pred, state_logit = model(x)
gaze_pred = gaze_pred.cpu().numpy()[0]
state_pred = (torch.sigmoid(state_logit) > 0.5).item()

print(f"Label:  gaze=({gaze_label[0]:.1f}, {gaze_label[1]:.1f}), state={'closed' if state_label else 'open'}")
print(f"Pred:   gaze=({gaze_pred[0]:.1f}, {gaze_pred[1]:.1f}), state={'closed' if state_pred else 'open'}")
print(f"Error:  {np.linalg.norm(gaze_pred - gaze_label):.2f} px")

# ============================================================
# Prepare event voxel for plotting
# ============================================================
voxel = sample['input']  # [10, 640, 480] float16 → already correct
voxel = voxel.cpu().numpy()

# Custom colormap: blue (positive), white (zero), red (negative)
cmap = plt.cm.bwr
cmap_colors = cmap(np.linspace(0, 1, 256))
cmap_colors[127:129] = [1, 1, 1, 1]  # force white center
custom_cmap = ListedColormap(cmap_colors)

# ============================================================
# PLOT 1. 2D XY (Last T_WINDOW from SELECTED_LABEL_N) 
# ============================================================
plt.figure(figsize=(15, 7))

plt.subplot(1, 2, 1)
sum_events = voxel.sum(axis=0)                    # accumulate all 10 bins
max_val = max(1, abs(sum_events).max())

xx, yy = np.where(sum_events != 0)
vals = sum_events[xx, yy]

sc = plt.scatter(xx, yy, c=vals, cmap=custom_cmap, s=6,
                 vmin=-max_val, vmax=max_val, edgecolors='none')
plt.colorbar(sc, label='Net Events (ON − OFF)')
plt.xlim(0, 640)
plt.ylim(0, 480)
# plt.gca().invert_yaxis()  # match image coordinates
plt.gca().set_aspect('equal')
plt.title(f"2D Event Accumulation (100 ms) — Sample {SAMPLE_IDX}")
plt.xlabel("X (px)")
plt.ylabel("Y (px)")

# Plot label and prediction
plt.plot(gaze_label[0], gaze_label[1], 'r+', markersize=12, markeredgewidth=3, label='Ground Truth')
plt.plot(gaze_pred[0], gaze_pred[1], 'g+', markersize=12, markeredgewidth=3, label='Prediction')
plt.legend()

# ============================================================
#  PLOT 2. 3D T-X-Y (Last T_WINDOW from SELECTED_LABEL_N, separates in frames for each BIN) 
# ============================================================
ax = plt.subplot(1, 2, 2, projection='3d')

for t in range(10):
    layer = voxel[t]
    xx, yy = np.where(layer != 0)
    if len(xx) == 0:
        continue
    zz = np.full_like(xx, t)
    vals = layer[xx, yy]
    ax.scatter(zz, xx, yy, c=vals, cmap='bwr', s=2, alpha=0.2, vmin=-1, vmax=1)

# Plot prediction and label at last time bin
ax.scatter([9], [gaze_pred[0]], [gaze_pred[1]], c='green', s=200, marker='+', linewidth=3, label='Pred')
ax.scatter([9], [gaze_label[0]], [gaze_label[1]], c='red', s=150, marker='+', linewidth=3, label='GT')

ax.set_xlabel("Time bin (0–9)")
ax.set_ylabel("X (px)")
ax.set_zlabel("Y (px)")
ax.set_title("3D Spatio-Temporal Event Voxel")
ax.legend()
ax.set_xlim(0, 9)
ax.set_ylim(0, 639)
ax.set_zlim(0, 479)

plt.suptitle(f"EyeTennSt Inference — {SPLIT}/{REC_ID} — Sample {SAMPLE_IDX} | "
             f"Error: {np.linalg.norm(gaze_pred - gaze_label):.2f} px | "
             f"Blink: {'Correct' if state_pred == state_label else 'Wrong'}", fontsize=14)

plt.tight_layout()
plt.show()