# 3_train.py
# Full training script for EyeTennSt on your 13 GB int8 dataset
# For input on the fly: reconstruct T=10 + convert to float16
# Input: [B, 10, 640, 480] float16 → 10 time bins (100 ms)
# Output: (x, y) gaze in pixels + eye state (0=open, 1=closed)

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import csv
from datetime import datetime
from pathlib import Path
import numpy as np
from tqdm import tqdm
from _2_3_model_f16_last_10_gradual import EyeTennSt
import signal
import time

# TF32 can spike power on Ampere/Turing cards
torch.backends.cuda.matmul.allow_tf32 = False

# Benchmark mode can cause huge power spikes at start
torch.backends.cudnn.benchmark = False

#
import os
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
os.environ["TORCH_LOGS"] = "" 
os.environ["TORCH_COMPILE_DEBUG"] = "0"
import warnings
warnings.filterwarnings("ignore", category=UserWarning, module="torch._dynamo")
warnings.filterwarnings("ignore", category=UserWarning, module="torch._logging")

# ============================================================
# CONFIGURATION — NOW OPTIMIZED FOR SPEED
# ============================================================
BATCH_SIZE = 8  
NUM_EPOCHS = 150
LEARNING_RATE = 0.004
WEIGHT_DECAY = 0.005
DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# Alienware paths
DATA_ROOT = Path("/home/dronelab-pc-1/Jon/IndustrialProject/akida_examples/1_ec_example/training/preprocessed_open")
LOG_DIR = Path(f"/home/dronelab-pc-1/Jon/IndustrialProject/akida_examples/1_ec_example/training/runs/tennst_2_11_8GB_b{BATCH_SIZE}_e{NUM_EPOCHS}_lr{LEARNING_RATE}")
LOG_DIR.mkdir(exist_ok=True)
LOG_FILE = LOG_DIR / f"training_log_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"

print(f"Training on {DEVICE}")
print(f"Batch size: {BATCH_SIZE}, Epochs: {NUM_EPOCHS}")

# ============================================================
# CUSTOM DATASET – builds 10-bin windows on-the-fly + float16
# ============================================================
class EyeTrackingDataset(Dataset):
    def __init__(self, split="train"):
        self.split = split
        self.recordings = [] # list of recordings (recording_dir, voxels, labels, num_frames_in_recording)
        self.samples = [] # list of samples (recording_index, frame_index)
        split_path = DATA_ROOT / split

        for rec_dir in sorted(split_path.iterdir()):
            if not rec_dir.is_dir():
                continue
            voxels_path = rec_dir / "voxels.pt"
            labels_path = rec_dir / "labels.txt"
            if not voxels_path.exists() or not labels_path.exists():
                continue

            # Memory-map → zero RAM usage until accessed
            voxels = torch.load(voxels_path, map_location="cpu", mmap=True, weights_only=True)  # [N, 640, 480] int8
            labels = np.loadtxt(labels_path, dtype=np.float32) # [N, 4]: t, x, y, state

            self.recordings.append({
                'rec_dir': rec_dir,
                'voxels': voxels,
                'labels': labels,
                'num_frames': len(voxels)
            })
        # Build flat index
        for rec_idx, item in enumerate(self.recordings):
            for i in range(item['num_frames']):
                self.samples.append((rec_idx, i))
        print(f"{split.upper()} dataset: {len(self.samples)} samples from {len(self.recordings)} recordings")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        rec_idx, frame_idx = self.samples[idx]
        item = self.recordings[rec_idx]
        # .to(torch.float16)
        voxels = item['voxels'].half()  # [N, 640, 480] int8 to float16 
        labels = item['labels']

        # Build 10-bin window
        start_idx = max(0, frame_idx - 9)
        window = voxels[start_idx:frame_idx + 1]
        if window.shape[0] < 10:
            pad = 10 - window.shape[0]
            padding = torch.zeros(pad, 640, 480, dtype=torch.float16)
            window = torch.cat([padding, window], dim=0)

        # Targets
        x, y = labels[frame_idx, 1], labels[frame_idx, 2]
        target = torch.tensor([x, y], dtype=torch.float32)

        return {
            'input': window,  # [10, 640, 480] float16
            'target': target,  # [2] float32
        }

# ============================================================
# SIMPLE LOGGER (Instead of Tensorboard)
# ============================================================
class SimpleLogger:
    def __init__(self, csv_path):
        self.csv_path = csv_path
        with open(csv_path, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(['epoch', 'train_loss_mse', 'val_loss_mse', 'lr', 'epoch_time_sec'])

    def log(self, epoch, train_loss, val_loss, lr, epoch_time):
        with open(self.csv_path, 'a', newline='') as f:
            writer = csv.writer(f)
            writer.writerow([epoch+1, f"{train_loss:.4f}", f"{val_loss:.4f}",
                           f"{lr:.6f}", f"{epoch_time:.2f}"])
        print(f"Epoch {epoch+1} | Val MSE: {val_loss:.2f}px | LR: {lr:.6f} | Time: {epoch_time:.2f}s")

# ============================================================
# MAIN TRAINING LOOP — NOW WITH SPEED BOOSTS
# ============================================================
def main():
    train_dataset = EyeTrackingDataset(split="train")
    val_dataset = EyeTrackingDataset(split="test")

    # FAST DATALOADER: num_workers=12, persistent_workers, prefetch_factor
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True,
                              num_workers=12, persistent_workers=True, prefetch_factor=4,
                              pin_memory=True, drop_last=True)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False,
                            num_workers=12, persistent_workers=True, prefetch_factor=4,
                            pin_memory=True)

    # Model
    torch.cuda.empty_cache()
    model = EyeTennSt(t_kernel_size=5, s_kernel_size=3, n_depthwise_layers=4).to(DEVICE)

    # DATA PARALLEL for multi-GPU setups
    if torch.cuda.device_count() > 1:
        print(f"Using {torch.cuda.device_count()} GPUs with DataParallel...")
        model = torch.nn.DataParallel(model)

    # TORCH.COMPILE — still great, now even more stable
    print("Compiling model with torch.compile()...")
    model = torch.compile(model, mode="reduce-overhead") # mode="default","max-autotune"

    criterion = nn.MSELoss()
    optimizer = optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)

    # OneCycleLR – super fast convergence
    total_steps = len(train_loader) * NUM_EPOCHS
    scheduler = optim.lr_scheduler.OneCycleLR(optimizer, max_lr=LEARNING_RATE,
                                              total_steps=total_steps, pct_start=0.025,
                                              anneal_strategy='cos')

    # MIXED PRECISION — HUGE memory + speed win
    scaler = torch.amp.GradScaler('cuda') 

    # Logging
    logger = SimpleLogger(LOG_FILE)
    print(f"Logging to: {LOG_FILE}")
    best_val_loss = float('inf')

    def save_on_interrupt(sig, frame):
        print("\nCtrl+C detected! Saving current model as 'stopped.pth'...")
        torch.save(model.state_dict(), f"{LOG_DIR}/stopped.pth")
        print("Model saved. Exiting.")
        exit(0)

    signal.signal(signal.SIGINT, save_on_interrupt)
    epoch_time = 0.0
    n_train = len(train_loader)
    n_val = len(val_loader)

    # Training loop
    for epoch in range(NUM_EPOCHS):
        model.train()
        train_loss_total = 0.0
        epoch_start_time = time.time()

        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{NUM_EPOCHS} [Train]")

        for batch in pbar:
            # model expects [B, T, W, H] float16
            x = batch['input'].to(DEVICE, non_blocking=True) 

            target = batch['target'].to(DEVICE, non_blocking=True)

            optimizer.zero_grad(set_to_none=True)  # faster than zero_grad()

            # MIXED PRECISION FORWARD + BACKWARD
            with torch.amp.autocast('cuda'):
                pred = model(x)
                loss = criterion(pred, target)

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            scheduler.step()

            # Metrics
            train_loss_total += loss.item()
            pbar.set_postfix({'loss': f"{loss.item():.3f}", 'lr': f"{scheduler.get_last_lr()[0]:.6f}"})

        epoch_time += time.time() - epoch_start_time

        logger.log(epoch,
                    train_loss_total / n_train,
                    0.0,  # val loss placeholder
                    scheduler.get_last_lr()[0],
                    epoch_time)
            
        epoch_time = 0.0

        # Validation every 5 epochs
        VAL_EPOCH_INTERVAL = 5
        if (epoch + 1) % VAL_EPOCH_INTERVAL == 0 or epoch == NUM_EPOCHS - 1:
            model.eval()
            val_loss_total = 0.0
            epoch_start_time = time.time()

            with torch.no_grad():
                for batch in val_loader:
                    x = batch['input'].to(DEVICE, non_blocking=True)
                    target = batch['target'].to(DEVICE, non_blocking=True)

                    with torch.amp.autocast('cuda'):
                        pred = model(x)
                        loss = criterion(pred, target)

                    val_loss_total += loss.item()

            epoch_time += time.time() - epoch_start_time

            # Logging
            logger.log(epoch,
                       train_loss_total / n_train,
                       val_loss_total / n_val,
                       scheduler.get_last_lr()[0],
                       epoch_time)
            
            epoch_time = 0.0

            if val_loss_total < best_val_loss:
                best_val_loss = val_loss_total
                torch.save(model.state_dict(), f"{LOG_DIR}/best.pth")
                print("New best model saved!")

    torch.save(model.state_dict(), f"{LOG_DIR}/final.pth")
    print("Training complete! Best model: best.pth")

if __name__ == "__main__":
    main()