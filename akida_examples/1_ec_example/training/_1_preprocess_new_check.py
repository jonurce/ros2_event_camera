# check_voxel.py
import torch
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import ListedColormap

# Config
SPLIT = "train"
REC= "1_5"

# Jetson paths
# PT_FILE = Path(f"/home/jetson/Jon/IndustrialProject/akida_examples/1_ec_example/training/preprocessed/{SPLIT}/{REC}/voxels.pt")
# LABEL_FILE = Path(f"/home/jetson/Jon/IndustrialProject/akida_examples/1_ec_example/training/preprocessed/{SPLIT}/{REC}/labels.txt")

# Alienware paths
PT_FILE = Path(f"/home/dronelab-pc-1/Jon/IndustrialProject/akida_examples/1_ec_example/training/preprocessed/{SPLIT}/{REC}/voxels.pt")
LABEL_FILE = Path(f"/home/dronelab-pc-1/Jon/IndustrialProject/akida_examples/1_ec_example/training/preprocessed/{SPLIT}/{REC}/labels.txt")

H, W = 480, 640
N_BINS = 10
T_LABELED = 10000 # 100Hz = 0.01s = 10000 us
T_WINDOW = T_LABELED * N_BINS
SELECTED_LABEL_N = 400

# ========================================
# LOAD DATA
# ========================================

# Load voxel
voxels = torch.load(PT_FILE)
print(f"Loaded: {PT_FILE.name}")
print(f"  Shape: {voxels.shape} → should be (N, {W}, {H})")

# Load label
labels = np.loadtxt(LABEL_FILE, dtype=np.int64) # [N, 4]: time, x, y, state
times, label_x_all, label_y_all, states = labels.T
print(f"Loaded: {LABEL_FILE.name}")
print(f'  Shape: {labels.shape} → should be (N, 4)')

# Safety check
assert len(voxels) == len(labels), "Mismatch between voxels and labels!"

# Get the selected label
t_selected = times[SELECTED_LABEL_N]
label_x = label_x_all[SELECTED_LABEL_N]
label_y = label_y_all[SELECTED_LABEL_N]
print(f"Selected label {SELECTED_LABEL_N}: time={t_selected}, gaze=({label_x}, {label_y})")

# ========================================
# Compute max values for accumulated events in T_WINDOW
# ========================================

max_all = 0
max_pos = 0
for index in range(len(voxels)):
    start_idx = max(0, index - N_BINS + 1)
    end_idx = index + 1
    window_voxels = voxels[start_idx:end_idx]
    if window_voxels.shape[0] < N_BINS:
        pad = N_BINS - window_voxels.shape[0]
        padding = torch.zeros(pad, W, H, device=window_voxels.device)
        window_voxels = torch.cat([padding, window_voxels], dim=0)

    # Count all events: +1 (ON), -1 (OFF)
    sum_all_events = window_voxels.sum(dim=0).cpu().numpy()
    if np.abs(sum_all_events).max() > max_all: max_all = np.abs(sum_all_events).max()

    # Only count +1 events (ON), ignore -1 (OFF)
    sum_pos_events = (window_voxels > 0).sum(dim=0).cpu().numpy()
    if np.abs(sum_pos_events).max() > max_pos: max_pos = np.abs(sum_pos_events).max()

print(f'Max pos & neg: {max_all}')
print(f'Max only pos: {max_pos}')


# ========================================
# Reconstruct the N_BINS voxel for the T_WINDOW before the selected label
# ========================================
start_idx = max(0, SELECTED_LABEL_N - N_BINS + 1)
end_idx = SELECTED_LABEL_N + 1
window_voxels = voxels[start_idx:end_idx] # [10, 640, 480] or less at beginning

# Pad if at the start of recording
if window_voxels.shape[0] < N_BINS:
    pad = N_BINS - window_voxels.shape[0]
    padding = torch.zeros(pad, W, H, device=window_voxels.device)
    window_voxels = torch.cat([padding, window_voxels], dim=0)

# Now: window_voxels is [10, 640, 480]
print(f"Window shape: {window_voxels.shape}")


# ========================================
# Custom colormap: white=0, red=neg, blue=pos
# ========================================
cmap = plt.cm.bwr  # blue-white-red
cmap_colors = cmap(np.linspace(0, 1, 256))
cmap_colors[127:129] = [1, 1, 1, 1]  # force center to white
custom_cmap = ListedColormap(cmap_colors)

# ========================================
# PLOT 1. 2D XY (Last T_WINDOW from SELECTED_LABEL_N) 
# ========================================
plt.figure(figsize=(14, 6))

plt.subplot(1, 2, 1)

# Count all events: +1 (ON), -1 (OFF)
sum_events = window_voxels.sum(dim=0).cpu().numpy()

# Only count +1 events (ON), ignore -1 (OFF)
# sum_events = (window_voxels > 0).sum(dim=0).cpu().numpy()

max_val = np.abs(sum_events).max()
xx, yy = np.where(sum_events != 0)
vals = sum_events[xx, yy]

plt.scatter(xx, yy, c=vals, cmap=custom_cmap, s=5, vmin=-max_val, vmax=max_val, edgecolors='none')
# plt.scatter(xx, yy, c=vals, cmap=custom_cmap, s=5, vmin=-5, vmax=5, edgecolors='none')
plt.gca().set_aspect('equal', adjustable='box')
plt.xlim(0, W)
plt.ylim(0, H)
plt.colorbar(label='Net Events (ON − OFF)')
plt.title(f'Sum of Events in [{t_selected - T_WINDOW:,} − {t_selected:,}] µs\n(Label {SELECTED_LABEL_N})')
plt.xlabel('X (0 → 639)')
plt.ylabel('Y (0 → 479)')

plt.plot(label_x, label_y, 'k+', markersize=18, markeredgewidth=3, label='Gaze Label')
plt.legend()


# ========================================
# PLOT 2. 3D T-X-Y (Last T_WINDOW from SELECTED_LABEL_N, separates in frames for each BIN) 
# ========================================
ax = plt.subplot(1, 2, 2, projection='3d')

for t_bin in range(N_BINS):
    layer = window_voxels[t_bin].cpu().numpy()
    xx, yy = np.where(layer != 0)  # yy = row index → Y, xx = col → X
    if len(xx) == 0:
        continue
    zz = np.full_like(xx, t_bin)
    vals = layer[xx, yy]
    ax.scatter(zz, xx, yy, c=vals, cmap='bwr', s=1, alpha=0.8, vmin=-1, vmax=1)

# Gaze point as line across time
# ax.plot([0, N_BINS-1], [label_x], [label_y], 'k+', linewidth=3)
ax.plot([N_BINS-1], [label_x], [label_y], 'k+', markersize=18, markeredgewidth=3)
# ax.scatter([N_BINS-1], [label_x], [label_y], 'k+', s=200,)

ax.set_ylabel('X')
ax.set_zlabel('Y')
ax.set_xlabel('Time bin (0–9)')
ax.set_title(f'3D Event Voxel ({T_WINDOW}us window separated in {N_BINS} bins, {T_LABELED}us each)')

ax.set_xlim(0, N_BINS-1)
ax.set_ylim(0, W-1)
ax.set_zlim(0, H-1)

plt.tight_layout()
plt.show()

