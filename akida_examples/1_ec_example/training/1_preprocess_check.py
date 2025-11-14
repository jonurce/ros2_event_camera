# check_voxel.py
import torch
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import ListedColormap

# Config
SPLIT = "train"
REC= "10_1"
SAMPLE = "000230"
PT_FILE = Path(f"/home/jetson/Jon/IndustrialProject/akida_examples/1_ec_example/training/preprocessed/{SPLIT}/{REC}/sample_{SAMPLE}.pt")
LABEL_FILE = Path(f"/home/jetson/Jon/IndustrialProject/akida_examples/1_ec_example/training/preprocessed/{SPLIT}/{REC}/sample_{SAMPLE}.label.txt")
H, W = 480, 640
T_BINS = 10

# Load
voxel = torch.load(PT_FILE)
print(f"Loaded: {PT_FILE.name}")
print(f"  Shape: {voxel.shape} → should be ({T_BINS}, {W}, {H})")
print(f"  Dtype: {voxel.dtype}")
print(f"  Device: {voxel.device}")
print(f"  Min: {voxel.min():.1f}, Max: {voxel.max():.1f}, Mean: {voxel.mean():.4f}")
print(f"  Non-zero events: {(voxel != 0).sum().item()}")

# Load label
with open(LABEL_FILE) as f:
    label_x, label_y, _ = map(int, f.read().split())

# === Custom colormap: white=0, red=neg, blue=pos ===
cmap = plt.cm.bwr  # blue-white-red
cmap_colors = cmap(np.linspace(0, 1, 256))
cmap_colors[127:129] = [1, 1, 1, 1]  # force center to white
custom_cmap = ListedColormap(cmap_colors)

# === 1. 2D XY (Last 10 ms) ===
plt.figure(figsize=(14, 6))

plt.subplot(1, 2, 1)

max = np.abs(voxel[-1]).max()
xx, yy = np.where(voxel[-1] != 0)
vals = voxel[-1][xx, yy]
plt.scatter(xx, H - 1 - yy, c=vals, cmap=custom_cmap, s=2, vmax=max, vmin=-max)
plt.gca().set_aspect('equal', adjustable='box')
plt.xlim(0, W)
plt.ylim(0, H)
plt.colorbar(label='Net Events (ON - OFF)')
plt.title('XY Event Polarity In Last 10ms')
plt.xlabel('X (0 → 639)')
plt.ylabel('Y (0 → 479)')
plt.grid(False)
plt.plot(label_x, H - 1 - label_y, 'k+', markersize=16, markeredgewidth=3)

# === 2. 3D T-X-Y ===
ax = plt.subplot(1, 2, 2, projection='3d')

for t_bin in range(T_BINS):
    xx, yy = np.where(voxel[t_bin] != 0)  # yy = row (Y), xx = col (X)
    if len(xx) == 0:
        continue
    zz = np.full_like(xx, t_bin)
    vals = voxel[t_bin][xx, yy]
    ax.scatter(zz, xx, H - 1 - yy, c=vals, cmap='bwr', s=2, alpha=0.8, vmin=-1, vmax=1)

# Fix 3D view for correct Y orientation
ax.set_xlim(0, T_BINS-1)
ax.set_ylim(0, W-1)
ax.set_zlim(0, H-1)
ax.set_xlabel('Time bin')
ax.set_ylabel('X')
ax.set_zlabel('Y')
ax.set_title('3D Voxel (T-X-Y)')

plt.tight_layout()
plt.show()

