# 2_train.py — ON-THE-FLY, WEIGHT DECAY, LIVE LOSS
# → Raw .h5 + label.txt → 1122-class gaze+blink

import h5py
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from pathlib import Path
import re
from tqdm import tqdm
import torch.nn as nn
import torch.optim as optim

# ========================================
# CONFIG
# ========================================
T_BINS = 10
WINDOW_MS = 100
H, W = 480, 640
DATA_ROOT = Path("/home/jetson/Jon/IndustrialProject/akida_examples/1_ec_example/training/event-based-eye-tracking-cvpr-2025/3ET+ dataset/event_data")
BATCH_SIZE = 4
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Training
EPOCHS = 20
LR = 1e-3
WEIGHT_DECAY = 1e-4  # L2 Regularization

# ========================================
# 1. ON-THE-FLY DATASET
# ========================================
class RawGazeDataset(Dataset):
    def __init__(self, split="train"):
        self.split_path = DATA_ROOT / split
        self.folders = sorted([f for f in self.split_path.iterdir() if f.is_dir()])
        self.samples = []

        print(f"Indexing {split} recordings...")
        for folder in self.folders:
            h5_file = folder / f"{folder.name}.h5"
            txt_file = folder / "label.txt"
            if not h5_file.exists() or not txt_file.exists():
                continue

            h5 = h5py.File(h5_file, 'r')
            events = h5['events']
            labels = self.load_labels(txt_file)
            if len(labels) == 0:
                h5.close()
                continue

            sort_idx = np.argsort(events['t'])
            events = events[sort_idx]

            t_labels_us = np.arange(len(labels)) * 10000
            for i in range(len(labels)):
                self.samples.append({
                    'h5': h5,
                    'events': events,
                    'label': labels[i],
                    't_label': t_labels_us[i]
                })

        print(f"{len(self.samples)} samples ready for streaming")

    def load_labels(self, txt_path):
        with open(txt_path) as f:
            lines = f.readlines()
        labels = []
        for line in lines:
            nums = re.findall(r'\d+', line)
            if len(nums) >= 3:
                x, y, state = map(int, nums[:3])
                labels.append([x, y, state])
        return np.array(labels, dtype=np.int32)

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        item = self.samples[idx]
        events = item['events']
        label = item['label']
        t_label = item['t_label']

        t_start = t_label - WINDOW_MS * 1000
        t_end = t_label

        t_events = events['t']
        left = np.searchsorted(t_events, t_start, side='left')
        right = np.searchsorted(t_events, t_end, side='right')

        if left >= right:
            voxel = np.zeros((T_BINS, H, W), dtype=np.float32)
        else:
            window = events[left:right]
            voxel = self.events_to_voxel(window, t_start, t_end)

        x_px = label[0]
        y_px = label[1]
        blink = float(label[2])
        target = torch.tensor([x_px, y_px, blink])

        return torch.tensor(voxel), target

    def events_to_voxel(self, events, t_start, t_end):
        voxel = np.zeros((T_BINS, H, W), dtype=np.float32)
        t = events['t']
        x = events['x']
        y = events['y']
        p = events['p']
        dt = t_end - t_start + 1e-6
        bin_idx = ((t - t_start) * T_BINS / dt).astype(np.int32)
        bin_idx = np.clip(bin_idx, 0, T_BINS - 1)
        pol = np.where(p == 1, 1.0, -1.0)
        np.add.at(voxel, (bin_idx, y, x), pol)
        return voxel

    def __del__(self):
        for item in self.samples:
            if 'h5' in item:
                try:
                    item['h5'].close()
                except:
                    pass

# ========================================
# 2. MODEL — 1122-CLASS CLASSIFICATION
# ========================================
N_CLASSES = W + H + 2  # x:0-639, y:640-1119, blink:1120-1121

class TennStClassify(nn.Module):
    def __init__(self):
        super().__init__()
        c = [1, 16, 32, 64, 128, 256]
        self.backbone = nn.Sequential(
            nn.Conv3d(c[0], c[1], (3,5,5), stride=(1,2,2), padding=(1,2,2)),
            nn.BatchNorm3d(c[1]), nn.ReLU(),
            nn.Conv3d(c[1], c[2], (3,5,5), stride=(1,2,2), padding=(1,2,2)),
            nn.BatchNorm3d(c[2]), nn.ReLU(),
            nn.Conv3d(c[2], c[3], (3,3,3), stride=(1,2,2), padding=(1,1,1)),
            nn.BatchNorm3d(c[3]), nn.ReLU(),
            nn.Conv3d(c[3], c[4], (3,3,3), stride=(1,2,2), padding=(1,1,1)),
            nn.BatchNorm3d(c[4]), nn.ReLU(),
            nn.Conv3d(c[4], c[5], (1,1,1)),
            nn.BatchNorm3d(c[5]), nn.ReLU(),
        )
        self.head = nn.Linear(c[5], N_CLASSES)

    def forward(self, x):
        x = x.unsqueeze(1)  # (B,1,10,480,640)
        x = self.backbone(x)
        x = x.mean([2,3,4])
        x = self.head(x)
        return x

# ========================================
# 3. TRAINING LOOP — WEIGHT DECAY + LIVE LOSS
# ========================================
def train():
    train_ds = RawGazeDataset("train")
    val_ds = RawGazeDataset("test")

    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, num_workers=2, pin_memory=True)
    val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=2, pin_memory=True)

    model = TennStClassify().to(DEVICE)
    criterion = nn.CrossEntropyLoss()
    
    optimizer = optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)

    print(f"\nStarting training")

    for epoch in range(EPOCHS):
        model.train()
        running_loss = 0.0
        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{EPOCHS}", leave=True)

        for step, (voxels, labels) in enumerate(pbar):
            voxels, labels = voxels.to(DEVICE), labels.to(DEVICE)

            # Convert raw pixels to class indices
            x_px, y_px, blink = labels.T
            x_class = x_px.long().clip(0, W - 1)
            y_class = y_px.long().clip(0, H - 1)
            blink_class = blink.long()

            # Build target: x + W*y + (W+H)*blink
            target = x_class + W * y_class + (W + H) * blink_class

            logits = model(voxels)
            loss = criterion(logits, target)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # LIVE LOSS PRINT
            running_loss += loss.item()
            if step % 10 == 0:
                pbar.set_postfix({
                    'loss': f'{loss.item():.4f}',
                    'avg': f'{running_loss/(step+1):.4f}'
                })

        print(f"Epoch {epoch+1} completed — Final avg loss: {running_loss/len(train_loader):.4f}")

    torch.save(model.state_dict(), "model_1122_wd.pth")
    print("\nTraining complete! Model saved as 'model_1122_wd.pth'")

# ========================================
# MAIN
# ========================================
if __name__ == "__main__":
    train()