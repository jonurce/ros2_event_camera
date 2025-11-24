# _3_4_train_fast_akida.py
# Training script for the EXACT Akida gaze model (T=50, 128×96, heatmap output)
# Input:  [B, 2, 50, 96, 128] uint8
# Target: [B, 3, 50, 3, 4]  → confidence + x_offset + y_offset
# Loss: L1 on soft-argmax gaze + CrossEntropy on confidence channel

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from pathlib import Path
from datetime import datetime
import csv
from tqdm import tqdm
import signal
from _2_5_model_uint8_akida import EyeTennSt 

# TF32 can spike power on Ampere/Turing cards
torch.backends.cuda.matmul.allow_tf32 = False

# Benchmark mode can cause huge power spikes at start
torch.backends.cudnn.benchmark = False

# ================================================
# CONFIG
# ================================================
BATCH_SIZE = 16    
NUM_EPOCHS = 200
LEARNING_RATE = 0.003
WEIGHT_DECAY = 0.01
DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

DATA_ROOT = Path("/home/dronelab-pc-1/Jon/IndustrialProject/akida_examples/1_ec_example/training/preprocessed_akida")
LOG_DIR = Path(f"/home/dronelab-pc-1/Jon/IndustrialProject/akida_examples/1_ec_example/training/runs/tennst_6_akida_b{BATCH_SIZE}_e{NUM_EPOCHS}")
LOG_DIR.mkdir(parents=True, exist_ok=True)
LOG_FILE = LOG_DIR / f"training_log_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"

print(f"Training Akida-exact model | Device: {DEVICE} | Batch: {BATCH_SIZE}")

# ================================================
# DATASET — loads uint8 voxels + float32 heatmaps
# ================================================
class AkidaGazeDataset(Dataset):
    def __init__(self, split="train"):
        # list of samples (recording_dir, frame_index)
        self.samples = []
        split_path = DATA_ROOT / split
        for rec_dir in sorted(split_path.iterdir()):
            if not rec_dir.is_dir():
                continue

            voxels_path = rec_dir / "voxels.pt"      # [N, 2, 50, 96, 128] uint8
            heatmaps_path = rec_dir / "heatmaps.pt"  # [N, 3, 50, 3, 4] float32

            if not voxels_path.exists() or not heatmaps_path.exists():
                continue

            # Memory-map → zero RAM usage until accessed
            # voxels = torch.load(voxels_path, map_location="cpu", mmap=True, weights_only=True)
            heatmaps = torch.load(heatmaps_path, map_location="cpu", mmap=True, weights_only=True)
            
            N = len(heatmaps)
            for i in range(N):
                self.samples.append((rec_dir, i))
        print(f"{split.upper()} → {len(self.samples)} samples")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        rec_dir, i = self.samples[idx]
        voxels = torch.load(rec_dir / "voxels.pt", map_location="cpu", mmap=True, weights_only=True)[i]      # [2,50,96,128] uint8
        heatmap = torch.load(rec_dir / "heatmaps.pt", map_location="cpu", mmap=True, weights_only=True)[i]  # [3,50,3,4] float32
        return {
            'input': voxels,     # stays uint8 until model does .float()
            'target': heatmap
        }

# ================================================
# LOSS — EXACTLY as in BrainChip Akida example
# ================================================
def akida_loss(pred, target):
    """
    pred:   [B, 3, 50, 3, 4] float32
    target: [B, 3, 50, 3, 4] float32
    Returns: L1 on soft-argmax gaze + CE on confidence
    """
    # 1. Confidence Cross-Entropy (over spatial dims only)
    conf_pred = pred[:, 0]      # [B, 50, 3, 4]
    conf_tgt  = target[:, 0]    # [B, 50, 3, 4]
    loss_ce = F.cross_entropy(conf_pred.flatten(2), conf_tgt.flatten(2), reduction='mean')

    # 2. L1 on soft-argmax gaze (only last time step, like original paper)
    last_pred = pred[:, :, -1]   # [B, 3, 3, 4]
    last_tgt  = target[:, :, -1] # [B, 3, 3, 4]

    prob = F.softmax(last_pred[:, 0], dim=(-2,-1))           # [B, 3, 4]
    x_pred = (prob * last_pred[:, 2].sigmoid()).sum(dim=[1,2])  # [B]
    y_pred = (prob * last_pred[:, 1].sigmoid()).sum(dim=[1,2])  # [B]

    x_gt = (prob.detach() * last_tgt[:, 2]).sum(dim=[1,2])
    y_gt = (prob.detach() * last_tgt[:, 1]).sum(dim=[1,2])

    loss_l1 = F.l1_loss(torch.stack([x_pred, y_pred], dim=1),
                        torch.stack([x_gt, y_gt], dim=1))

    return loss_ce + 10.0 * loss_l1, loss_ce.item(), loss_l1.item()

# ================================================
# MAIN TRAINING LOOP
# ================================================
def main():
    train_ds = AkidaGazeDataset("train")
    val_ds   = AkidaGazeDataset("test")

    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True,
                              num_workers=12, persistent_workers=True, pin_memory=True)
    val_loader   = DataLoader(val_ds,   batch_size=BATCH_SIZE, shuffle=False,
                              num_workers=12, persistent_workers=True, pin_memory=True)

    model = EyeTennSt(t_kernel_size=5, s_kernel_size=3, n_depthwise_layers=4).to(DEVICE)

    # Multi-GPU if available
    if torch.cuda.device_count() > 1:
        print(f"→ Using {torch.cuda.device_count()} GPUs with DataParallel")
        model = torch.nn.DataParallel(model)

    # torch.compile for max speed
    model = torch.compile(model, mode="reduce-overhead")

    optimizer = optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
    scheduler = optim.lr_scheduler.OneCycleLR(optimizer, max_lr=LEARNING_RATE,
                                              total_steps=len(train_loader)*NUM_EPOCHS,
                                              pct_start=0.1)

    scaler = torch.amp.GradScaler('cuda')  # mixed precision

    # Logger
    with open(LOG_FILE, 'w', newline='') as f:
        w = csv.writer(f)
        w.writerow(['epoch','train_loss','train_ce','train_l1','val_loss','val_mae_px','lr'])

    best_mae = float('inf')

    def save_sigint(sig, frame):
        torch.save(model.state_dict(), LOG_DIR / "interrupted.pth")
        print("\nSaved & exiting.")
        exit(0)
    signal.signal(signal.SIGINT, save_sigint)

    for epoch in range(NUM_EPOCHS):
        # === TRAIN ===
        model.train()
        train_loss = train_ce = train_l1 = 0.0
        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{NUM_EPOCHS}")
        for batch in pbar:
            x = batch['input'].to(DEVICE, non_blocking=True)      # uint8 → model does .float()
            y = batch['target'].to(DEVICE, non_blocking=True)

            optimizer.zero_grad(set_to_none=True)
            with torch.amp.autocast('cuda'):
                pred = model(x)                               # [B,3,50,3,4]
                loss, ce, l1 = akida_loss(pred, y)

            # main loss

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            scheduler.step()

            train_loss += loss.item()
            train_ce += ce
            train_l1 += l1
            pbar.set_postfix({'loss': f"{loss.item():.3f}", 'lr': f"{scheduler.get_last_lr()[0]:.2e}"})

        # === VALIDATION ===
        model.eval()
        val_loss = val_mae_px = 0.0
        with torch.no_grad():
            for batch in val_loader:
                x = batch['input'].to(DEVICE, non_blocking=True)
                y = batch['target'].to(DEVICE, non_blocking=True)
                with torch.amp.autocast('cuda'):
                    pred = model(x)
                    loss, _, l1 = akida_loss(pred, y)
                    # Convert L1 from [0,1]→[0,4) back to pixels
                    mae_px = l1 * 640 / 4.0   # W_OUT = 4 → scale to 640px
                val_loss += loss.item()
                val_mae_px += mae_px

        n = len(val_loader)
        val_mae_px /= n

        # Log
        with open(LOG_FILE, 'a', newline='') as f:
            w = csv.writer(f)
            w.writerow([epoch+1,
                        f"{train_loss/n:.4f}",
                        f"{train_ce/n:.4f}",
                        f"{train_l1/n:.4f}",
                        f"{val_loss/n:.4f}",
                        f"{val_mae_px:.2f}",
                        f"{scheduler.get_last_lr()[0]:.2e}"])

        print(f"→ Val MAE: {val_mae_px:.2f}px | Best: {best_mae:.2f}px")
        if val_mae_px < best_mae:
            best_mae = val_mae_px
            torch.save(model.state_dict(), LOG_DIR / "best.pth")
            print("  NEW BEST!")

    torch.save(model.state_dict(), LOG_DIR / "final.pth")
    print(f"\nTraining finished! Best MAE: {best_mae:.2f}px")

if __name__ == "__main__":
    main()