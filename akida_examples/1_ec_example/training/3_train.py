# 2_train.py
# → Train TennSt on 100ms causal voxels → 1122-class output
# → Output: gaze (x,y) + blink via classification

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np
from pathlib import Path
from tqdm import tqdm
import os

# ========================================
# CONFIG
# ========================================
T_BINS = 10
H, W = 480, 640
N_X = W      # 640 classes
N_Y = H      # 480 classes
N_BLINK = 2  # open/closed
N_CLASSES = N_X + N_Y + N_BLINK  # 1122

BATCH_SIZE = 4
EPOCHS = 20
LR = 1e-3
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


DATA_DIR = Path("/home/jetson/Jon/IndustrialProject/akida_examples/1_ec_example/training/preprocessed")


MODEL_DIR = Path("models")
MODEL_DIR.mkdir(exist_ok=True)

# ========================================
# 1. DATASET
# ========================================
    
class GazeDataset(Dataset):
    def __init__(self, split="train"):
        self.files = sorted((DATA_DIR / split).glob("*_voxels.npy"))
        self.samples = []
        for f in self.files:
            voxels = np.load(f, mmap_mode='r')
            labels = np.load(f.with_name(f.name.replace("_voxels", "_labels")), mmap_mode='r')
            for i in range(len(voxels)):
                self.samples.append((str(f), i))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        file_path, sample_idx = self.samples[idx]
        voxels = np.load(file_path, mmap_mode='r')[sample_idx]
        labels = np.load(file_path.replace("_voxels", "_labels"), mmap_mode='r')[sample_idx]
        return torch.tensor(voxels.copy()), torch.tensor(labels)

# ========================================
# 2. MODEL — TennStMini Classification
# ========================================
class SpatioTemporalBlock(nn.Module):
    def __init__(self, in_c, out_c, t_kernel=3, s_kernel=3, stride=1, depthwise=True):
        super().__init__()
        pad_t = t_kernel // 2
        pad_s = s_kernel // 2
        groups = in_c if depthwise else 1

        self.conv = nn.Conv3d(
            in_c, out_c, (t_kernel, s_kernel, s_kernel),
            stride=(1, stride, stride), padding=(pad_t, pad_s, pad_s),
            groups=groups, bias=False
        )
        self.bn = nn.BatchNorm3d(out_c)
        self.relu = nn.ReLU(inplace=True)

        if depthwise:
            self.pointwise = nn.Conv3d(in_c, out_c, 1, bias=False)

    def forward(self, x):
        if hasattr(self, 'pointwise'):
            x = self.pointwise(x)
        x = self.conv(x)
        x = self.bn(x)
        x = self.relu(x)
        return x

class TennStClassify(nn.Module):
    def __init__(self):
        super().__init__()
        channels = [1, 16, 32, 64, 128, 256]
        self.backbone = nn.Sequential(
            SpatioTemporalBlock(channels[0], channels[1], t_kernel=3, s_kernel=5, stride=2, depthwise=True),
            SpatioTemporalBlock(channels[1], channels[2], t_kernel=3, s_kernel=5, stride=2, depthwise=True),
            SpatioTemporalBlock(channels[2], channels[3], t_kernel=3, s_kernel=3, stride=2, depthwise=True),
            SpatioTemporalBlock(channels[3], channels[4], t_kernel=3, s_kernel=3, stride=2, depthwise=True),
            SpatioTemporalBlock(channels[4], channels[5], t_kernel=1, s_kernel=1, stride=1, depthwise=False),
        )
        self.head = nn.Linear(channels[-1], N_CLASSES)

    def forward(self, x):
        # x: (B, 10, 480, 640) → add channel dim
        x = x.unsqueeze(1)  # (B, 1, 10, 480, 640)
        x = self.backbone(x)  # (B, 256, T', H', W')
        x = x.mean([2, 3, 4])  # Global avg pool
        x = self.head(x)
        return x  # (B, 1122)

# ========================================
# 3. POST-PROCESSING: SOFT VOTING
# ========================================
def decode_prediction(logits):
    probs = torch.softmax(logits, dim=1).cpu().numpy()

    batch_x, batch_y, batch_blink = [], [], []
    for p in probs:
        # X: classes 0..639
        x_probs = p[:N_X]
        x_bin = np.argmax(x_probs)
        x_weighted = np.sum(x_probs * np.arange(N_X)) / (np.sum(x_probs) + 1e-8)
        batch_x.append(x_weighted / (N_X - 1))

        # Y: classes 640..1119
        y_probs = p[N_X:N_X+N_Y]
        y_bin = np.argmax(y_probs)
        y_weighted = np.sum(y_probs * np.arange(N_Y)) / (np.sum(y_probs) + 1e-8)
        batch_y.append(y_weighted / (N_Y - 1))

        # Blink: classes 1120, 1121
        blink_prob = p[-1]  # closed
        batch_blink.append(blink_prob)

    return np.array(batch_x), np.array(batch_y), np.array(batch_blink)

# ========================================
# 4. METRICS
# ========================================
def compute_metrics(preds, gts):
    x_pred, y_pred, blink_pred = preds
    x_gt, y_gt, blink_gt = gts

    # Gaze error in pixels
    err_x = np.abs(x_pred * (W - 1) - x_gt * (W - 1))
    err_y = np.abs(y_pred * (H - 1) - y_gt * (H - 1))
    l2_error = np.sqrt(err_x**2 + err_y**2)

    # 5px accuracy
    acc_5px = np.mean(l2_error <= 5.0)

    # Blink F1
    blink_pred_bin = (blink_pred > 0.5).astype(int)
    blink_gt_bin = blink_gt.astype(int)
    tp = np.sum((blink_pred_bin == 1) & (blink_gt_bin == 1))
    fp = np.sum((blink_pred_bin == 1) & (blink_gt_bin == 0))
    fn = np.sum((blink_pred_bin == 0) & (blink_gt_bin == 1))
    precision = tp / (tp + fp + 1e-8)
    recall = tp / (tp + fn + 1e-8)
    f1 = 2 * precision * recall / (precision + recall + 1e-8)

    return {
        "L2": l2_error.mean(),
        "5px": acc_5px,
        "Blink F1": f1
    }

# ========================================
# 5. TRAINING LOOP
# ========================================
def train():
    train_ds = GazeDataset("train")
    val_ds = GazeDataset("test")
    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, num_workers=4, pin_memory=True)
    val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=4, pin_memory=True)

    model = TennStClassify().to(DEVICE)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=LR)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS)

    best_l2 = float("inf")

    for epoch in range(EPOCHS):
        model.train()
        train_pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{EPOCHS} [Train]")
        for voxels, targets in train_pbar:
            voxels, targets = voxels.to(DEVICE), targets.to(DEVICE)
            logits = model(voxels)
            loss = criterion(logits, targets)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            train_pbar.set_postfix(loss=loss.item())

        # Validation
        model.eval()
        all_preds, all_gts = [], []
        with torch.no_grad():
            for voxels, targets in val_loader:
                voxels = voxels.to(DEVICE)
                logits = model(voxels)
                x_pred, y_pred, blink_pred = decode_prediction(logits)
                all_preds.append((x_pred, y_pred, blink_pred))
                # Recover GT
                idx = targets.numpy()
                x_gt = (idx % N_X) / (N_X - 1)
                y_gt = ((idx - (idx % N_X)) // N_X % N_Y) / (N_Y - 1)
                blink_gt = (idx >= N_X + N_Y).astype(float)
                all_gts.append((x_gt, y_gt, blink_gt))

        # Concat
        preds = [np.concatenate([p[i] for p in all_preds]) for i in range(3)]
        gts = [np.concatenate([g[i] for g in all_gts]) for i in range(3)]
        metrics = compute_metrics(preds, gts)

        print(f"\nVal L2: {metrics['L2']:.2f}px | 5px: {metrics['5px']:.1%} | Blink F1: {metrics['Blink F1']:.3f}")

        # Save best
        if metrics["L2"] < best_l2:
            best_l2 = metrics["L2"]
            torch.save(model.state_dict(), MODEL_DIR / "best_classification.pth")
            print("Best model saved!")

        scheduler.step()

    print("\nTraining complete!")
    print("Next: python 3_quantize.py")

# ========================================
# 6. MAIN
# ========================================
if __name__ == "__main__":
    train()