# _7_1_test_qat_plot.py
# Visualize a single sample using the QAT-quantized (int8/int4) EyeTennSt model
# → Identical to 4_1_test_tennsts_plot.py but loads quantized weights properly

import torch
import torch.nn as nn
from torch.utils.data import Dataset
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from training._2_3_model_f16_last_10_gradual import EyeTennSt  # ← your final model with gradual head
from training._4_1_test_tennst_plot import SingleRecordingDataset  # ← reuse dataset class

# ============================================================
# CONFIG — CHANGE THESE
# ============================================================
DEVICE = torch.device("cuda:0" if torch.cuda.device_count() > 0 else "cpu")

# Dataset
DATA_ROOT = Path("/home/dronelab-pc-1/Jon/IndustrialProject/akida_examples/1_ec_example/training/preprocessed_fast")

# Choose QAT model
BITS = 8

QAT_DIR = Path(f"quantized_models/qat_int{BITS}")
MODEL_PATH = QAT_DIR / "best_qat.pth"  # or "final_int{BITS}_qat.pth"

# Sample to visualize
SPLIT = "test"       # or "train"
REC_ID = "1_1"       # change as needed
SAMPLE_IDX = 500     # change as needed

print(f"Visualizing QAT {BITS}-bit model")
print(f"Model: {MODEL_PATH.name}")
print(f"Sample: {SPLIT}/{REC_ID} — index {SAMPLE_IDX}")

# ============================================================
# Load QAT-quantized model correctly
# ============================================================
# 1. Create base model
model = EyeTennSt(t_kernel_size=5, s_kernel_size=3, n_depthwise_layers=6)

# Support multi-GPU if available
if torch.cuda.device_count() > 1:
    print(f"Using {torch.cuda.device_count()} GPUs")
    model = torch.nn.DataParallel(model)

# TORCH.COMPILE — still great, now even more stable
print("Compiling model with torch.compile()...")
model = torch.compile(model, mode="reduce-overhead")

# 2. Apply quantization configuration (same as during QAT)
if BITS == 8:
    model.qconfig = torch.ao.quantization.get_default_qconfig('fbgemm')
else:
    model.qconfig = torch.ao.quantization.get_default_qat_qconfig('fbgemm', version=4)

# 3. Prepare + convert → creates quantized model structure
model_prepared = torch.ao.quantization.prepare(model.train())        # needed for structure
model_qat = torch.ao.quantization.convert(model_prepared.eval())  # final quantized model

# 4. Load the actual QAT weights
model_qat.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
model_qat.eval()
model_qat.to(DEVICE)

print(f"QAT {BITS}-bit model loaded successfully")

# ============================================================
# Load selected sample
# ============================================================
rec_path = DATA_ROOT / SPLIT / REC_ID
dataset = SingleRecordingDataset(rec_path)
sample = dataset[SAMPLE_IDX]

x = sample['input'].unsqueeze(0).to(DEVICE)        # [1, 10, 640, 480]
gaze_label = sample['gaze'].numpy()
state_label = int(sample['state'].item())

# ============================================================
# Inference with QAT model
# ============================================================
with torch.no_grad(), torch.cuda.amp.autocast():
    gaze_pred, state_logit = model_qat(x)

gaze_pred = gaze_pred.cpu().numpy()[0]
state_pred = (torch.sigmoid(state_logit) > 0.5).item()

print(f"Label:  gaze=({gaze_label[0]:.1f}, {gaze_label[1]:.1f}), state={'closed' if state_label else 'open'}")
print(f"Pred:   gaze=({gaze_pred[0]:.1f}, {gaze_pred[1]:.1f}), state={'closed' if state_pred else 'open'}")
print(f"Error:  {np.linalg.norm(gaze_pred - gaze_label):.2f} px")

# ============================================================
# Prepare voxel for plotting (same as before)
# ============================================================
voxel = sample['input'] # [10, 640, 480]

# Custom colormap
cmap = plt.cm.bwr
cmap_colors = cmap(np.linspace(0, 1, 256))
cmap_colors[127:129] = [1, 1, 1, 1]
custom_cmap = ListedColormap(cmap_colors)

# ============================================================
# PLOT 1: 2D accumulated events
# ============================================================
plt.figure(figsize=(15, 7))
plt.subplot(1, 2, 1)
sum_events = voxel.sum(axis=0)
max_val = max(1, abs(sum_events).max())
xx, yy = np.where(sum_events != 0)
vals = sum_events[xx, yy]

plt.scatter(xx, yy, c=vals, cmap=custom_cmap, s=6, vmin=-max_val, vmax=max_val, edgecolors='none')
plt.colorbar(label='Net Events (ON − OFF)')
plt.xlim(0, 640)
plt.ylim(0, 480)
plt.gca().set_aspect('equal')
plt.title(f"2D Event Accumulation (100 ms) — QAT {BITS}-bit — Sample {SAMPLE_IDX}")
plt.xlabel("X (px)")
plt.ylabel("Y (px)")

# Ground truth (red) and prediction (green)
plt.plot(gaze_label[0], gaze_label[1], 'r+', markersize=12, markeredgewidth=3, label='Ground Truth')
plt.plot(gaze_pred[0], gaze_pred[1], 'g+', markersize=12, markeredgewidth=3, label='QAT Prediction')
plt.legend()

# ============================================================
# PLOT 2: 3D spatio-temporal voxel
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

# Final time bin: prediction and label
ax.scatter([9], [gaze_pred[0]], [gaze_pred[1]], c='green', s=200, marker='+', linewidth=2, label='Pred')
ax.scatter([9], [gaze_label[0]], [gaze_label[1]], c='red', s=200, marker='+', linewidth=3, label='GT')

ax.set_xlabel("Time bin (0–9)")
ax.set_ylabel("X (px)")
ax.set_zlabel("Y (px)")
ax.set_title("3D Event Voxel — QAT Model")
ax.legend()
ax.set_xlim(0, 9)
ax.set_ylim(0, 639)
ax.set_zlim(0, 479)

plt.suptitle(f"QAT {BITS}-bit EyeTennSt — {SPLIT}/{REC_ID} — Sample {SAMPLE_IDX}\n"
             f"Error: {np.linalg.norm(gaze_pred - gaze_label):.2f} px | "
             f"Blink: {'Correct' if state_pred == state_label else 'Wrong'}", fontsize=14)

plt.tight_layout()
plt.show()