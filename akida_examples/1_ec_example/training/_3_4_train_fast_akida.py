# _3_4_train_fast_akida.py
# Training script for the EXACT Akida gaze model (T=50, 128×96, heatmap output)
# Input:  [B, 2, 50, 96, 128] uint8
# Target: [B, 3, 50, 3, 4]  → confidence + x_offset + y_offset
# Loss: L1 on soft-argmax gaze + CrossEntropy on confidence channel

import torch
import numpy as np
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

import os
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
os.environ["TORCH_LOGS"] = "" 
os.environ["TORCH_COMPILE_DEBUG"] = "0"
import warnings
warnings.filterwarnings("ignore", category=UserWarning, module="torch._dynamo")
warnings.filterwarnings("ignore", category=UserWarning, module="torch._logging")


# ================================================
# CONFIG
# ================================================
BATCH_SIZE = 128   
NUM_EPOCHS = 10
LEARNING_RATE = 0.002
WEIGHT_DECAY = 0.005
DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

DATA_ROOT = Path("/home/dronelab-pc-1/Jon/IndustrialProject/akida_examples/1_ec_example/training/preprocessed_akida")
LOG_DIR = Path(f"/home/dronelab-pc-1/Jon/IndustrialProject/akida_examples/1_ec_example/training/runs/tennst_12_akida_b{BATCH_SIZE}_e{NUM_EPOCHS}")
LOG_DIR.mkdir(parents=True, exist_ok=True)
LOG_FILE = LOG_DIR / f"training_log_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"

# From original data
W = 640
H = 480

# Akida model input/output sizes
W_IN = 128
H_IN = 96
W_OUT = 4
H_OUT = 3

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

        # [2,50,96,128] uint8
        voxels = torch.load(rec_dir / "voxels.pt", map_location="cpu", mmap=True, weights_only=True)[i]  
        
        # [3,50,3,4] float32   
        heatmap = torch.load(rec_dir / "heatmaps.pt", map_location="cpu", mmap=True, weights_only=True)[i]  
        
        return {
            'input': voxels,  
            'target': heatmap
        }

# ============================================================
# Visualize the RAW winning cell + offset (not soft-argmax)
# ============================================================
def get_out_gaze_point(pred):
    """
    pred: [B, 3, 50, 3, 4] (from model output)
    Returns: x_out, y_out (in [0,4) and [0,3)), cell_idx (y_idx, x_idx), max_conf
    """
    last = pred[:, :, -1]   # [B, 3, 3, 4]
    conf = last[:, 0]       # [B, 3, 4]

    B, H, W = conf.shape
    conf_flat = conf.reshape(B, -1)             # [B, 12]
    max_idx = conf_flat.argmax(dim=-1)          # [B]

    # Convert flat index → (y_idx, x_idx)
    y_idx = max_idx // W_OUT      # integer division
    x_idx = max_idx % W_OUT

    # Gather offsets from the winning cells
    # last[:,1] = y_offset raw, last[:,2] = x_offset raw
    y_offset = torch.gather(last[:, 1].reshape(B, -1), 1, max_idx.unsqueeze(1)).squeeze(1).sigmoid()
    x_offset = torch.gather(last[:, 2].reshape(B, -1), 1, max_idx.unsqueeze(1)).squeeze(1).sigmoid()

    
    # Final normalized gaze (in grid space: x ∈ [0,4), y ∈ [0,3))
    x_out = x_idx.float() + x_offset
    y_out = y_idx.float() + y_offset

    max_conf = conf_flat.gather(1, max_idx.unsqueeze(1)).squeeze(1)

    return x_out, y_out, (y_idx, x_idx), max_conf

# ============================================================
# Test get_raw_gaze_point()
# ============================================================
def test_get_raw_gaze_point():

    B, T, C, H, W_OUT = 4, 50, 3, 3, 4

    # TEST: Create fake heatmaps with known ground truth
    torch.manual_seed(42)
    pred = torch.randn(B, C, T, H, W_OUT) * 0.1  # small noise

    # Force known winning cells and offsets for each batch
    true_cells = [(1, 2), (0, 0), (2, 3), (1, 1)]   # (y_idx, x_idx)
    true_offsets = [(0.7, 0.3), (0.1, 0.9), (0.5, 0.5), (0.8, 0.2)]

    for b in range(B):
        y_idx, x_idx = true_cells[b]
        y_off, x_off = true_offsets[b]
        
        # Set high confidence at known cell
        pred[b, 0, :, y_idx, x_idx] = 1.0  # high logit → wins
        pred[b, 1, -1, y_idx, x_idx] = torch.logit(torch.tensor(y_off))  # y_offset raw
        pred[b, 2, -1, y_idx, x_idx] = torch.logit(torch.tensor(x_off))  # x_offset raw

    # Run function
    x_out, y_out, (y_idx, x_idx), max_conf = get_raw_gaze_point(pred)

    # Expected values
    expected_x = torch.tensor([2 + 0.3, 0 + 0.9, 3 + 0.5, 1 + 0.2])
    expected_y = torch.tensor([1 + 0.7, 0 + 0.1, 2 + 0.5, 1 + 0.8])

    print("Batch | X Out | Y Out | X Expected | Y Expected | Cell (y,x) | OK?")
    print("-" * 60)
    for b in range(B):
        ok = (abs(x_out[b] - expected_x[b]) < 1e-4) and (abs(y_out[b] - expected_y[b]) < 1e-4)
        print(f"{b:5} | {x_out[b]:.4f} | {y_out[b]:.4f} | {expected_x[b]:.4f} | {expected_y[b]:.4f} | "
            f"{y_idx[b].item(), x_idx[b].item()}     | {'YES' if ok else 'NO'}")

    # Final check
    assert torch.allclose(x_out, expected_x, atol=1e-4)
    assert torch.allclose(y_out, expected_y, atol=1e-4)
    print("\nAll tests passed! Function is 100% correct.")

# ============================================================
#  NOT USED * SOFT-ARGMAX GAZE EXTRACTION FOR VALIDATION LOSS
# ============================================================
def extract_gaze(pred):
    """
    pred: [B, 3, 50, 3, 4] → we only use last time step
    Returns: gaze error in normalized [0,1] → then scaled to pixels
    """
    last = pred[:, :, -1]                     # [B, 3, 3, 4]
    conf = last[:, 0]                         # [B, 3, 4]
    prob = F.softmax(conf.flatten(-2), dim=-1).view_as(conf)   # [B, 3, 4]

    x_offset = last[:, 2].sigmoid()           # [B, 3, 4]
    y_offset = last[:, 1].sigmoid()           # [B, 3, 4]

    x_pred = (prob * x_offset).sum(dim=[-2, -1])   # [B]
    y_pred = (prob * y_offset).sum(dim=[-2, -1])   # [B]

    return torch.stack([x_pred, y_pred], dim=1)    # [B, 2] in [0,1]

# ================================================
# LOSS — EXACTLY as in BrainChip Akida example
# ================================================
def akida_loss(pred, target):
    """
    pred:   [B, 3, 50, 3, 4] float32
    target: [B, 3, 50, 3, 4] float32
    Returns: L2 on soft-argmax gaze + CE on confidence
    """
    # 1. Confidence Cross-Entropy (over spatial dims only)
    # loss_ce (Cross-Entropy on confidence) -> “Put the mass on the correct 3×4 cell”
    conf_pred = pred[:, 0]      # [B, 50, 3, 4] → confidence map (logits)
    conf_tgt  = target[:, 0]    # [B, 50, 3, 4] → one-hot: 1.0 at true cell, 0 elsewhere
    loss_ce = F.cross_entropy(conf_pred.flatten(2), conf_tgt.flatten(2), reduction='mean')

    # 2. L1 on soft-argmax gaze (only last time step)
    # loss_l1 (L1 on soft-argmax gaze) -> “Once you know the cell, give me sub-pixel accuracy”
    # gaze_pred_norm = extract_gaze(pred) # [B,2] in [0,1]
    # gaze_gt_norm = extract_gaze(target)  # [B,2] in [0,1]
    # loss_l1 = F.l1_loss(gaze_pred_norm, gaze_gt_norm)

    pred_x, pred_y, pred_cell, pred_conf = get_out_gaze_point(pred)
    gt_x, gt_y, gt_cell, gt_conf = get_out_gaze_point(target)

    # L1 error in normalized space
    loss_l1 = F.l1_loss(
        torch.stack([pred_x / W_OUT, pred_y / H_OUT], dim=1),
        torch.stack([gt_x   / W_OUT, gt_y   / H_OUT], dim=1)
    )
    # or Euclidean in OUT pixel space
    loss_l2 = torch.sqrt(  (pred_x - gt_x)**2  +  (pred_y - gt_y)**2  ).mean()
    
    weight_l2 = 2000

    # return loss_ce + weight_l1 * loss_l1, loss_ce, loss_l1, weight_l1
    return loss_ce + weight_l2 * loss_l2, loss_ce, loss_l2, weight_l2

# ================================================
# MAIN TRAINING LOOP
# ================================================
def main():
    train_ds = AkidaGazeDataset("train")
    val_ds   = AkidaGazeDataset("test")

    # FAST DATALOADER: num_workers=12, persistent_workers, prefetch_factor
    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True,
                              num_workers=12, persistent_workers=True, prefetch_factor=4, 
                              pin_memory=True)
    val_loader   = DataLoader(val_ds,   batch_size=BATCH_SIZE, shuffle=False,
                              num_workers=12, persistent_workers=True, prefetch_factor=4,
                              pin_memory=True)

    # Model
    torch.cuda.empty_cache()
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
                                              pct_start=0.025, anneal_strategy='cos')

    # mixed precision
    scaler = torch.amp.GradScaler('cuda')  

    # Logger
    with open(LOG_FILE, 'w', newline='') as f:
        w = csv.writer(f)
        w.writerow(['epoch','train_loss','train_ce','train_l2','val_loss','val_loss_l2','lr'])

    # Initialize best metrics for validation and saving best model
    best_val_loss = float('inf')
    best_val_loss_l2 = float('inf')

    # Handle Ctrl+C
    def save_sigint(sig, frame):
        torch.save(model.state_dict(), LOG_DIR / "interrupted.pth")
        print("\nSaved & exiting.")
        exit(0)
    signal.signal(signal.SIGINT, save_sigint)

    # Training loop
    for epoch in range(NUM_EPOCHS):
        model.train()
        train_loss = train_loss_ce = train_loss_l2 = 0.0
        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{NUM_EPOCHS}")

        for batch in pbar:
            # Model expects input:[B,2,50,96,128] uint8
            x = batch['input'].to(DEVICE, non_blocking=True)

            # Models predicts: [B,3,50,3,4] float32
            y = batch['target'].to(DEVICE, non_blocking=True)

            optimizer.zero_grad(set_to_none=True)

            # Mixed precision forward and loss
            with torch.amp.autocast('cuda'):
                # Models predicts: [B,3,50,3,4] float32
                pred = model(x)                               
                loss, loss_ce, loss_l2, w_l2 = akida_loss(pred, y)

            # main loss backward with scaler
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            scheduler.step()

            # Metrics
            train_loss += loss.item()
            train_loss_ce += loss_ce.item()
            train_loss_l2 += loss_l2.item()
            
            pbar.set_postfix({
                'Total Loss': f"{loss.item():.3f}",
                'Loss CE': f"{loss_ce.item():.3f}",
                'Loss L2': f"{loss_l2.item():.3f}",
                'L2 Weight': f"{w_l2:.3f}",
                'Learning Rate': f"{scheduler.get_last_lr()[0]:.2e}"
            })

        # Validation every 5 epochs
        if (epoch + 1) % 5 == 0 or epoch == NUM_EPOCHS - 1:
            model.eval()
            val_loss = val_loss_l2 = 0.0
            with torch.no_grad():
                for batch in val_loader:
                    # Model expects input:[B,2,50,96,128] uint8
                    x = batch['input'].to(DEVICE, non_blocking=True)

                    # Models predicts: [B,3,50,3,4] float32
                    y = batch['target'].to(DEVICE, non_blocking=True)

                    with torch.amp.autocast('cuda'):
                        # Models predicts: [B,3,50,3,4] float32
                        pred = model(x)
                        loss, _, loss_l2, w_l2 = akida_loss(pred, y)
                        # Convert L1 from [0,1]→[0,4) back to pixels
                        # same as doing: mae_px = loss_l1 * (W / W_OUT)
                        # mae_px = loss_l1.item() * (W_IN + H_IN) / (W_OUT + H_OUT)
                    
                    val_loss += loss.item()
                    val_loss_l2 += loss_l2.item()

            n_train = len(train_loader)
            n_val   = len(val_loader)

            train_loss /= n_train
            train_loss_ce /= n_train
            train_loss_l2 /= n_train
            val_loss /= n_val
            val_loss_l2 /= n_val

            # Log
            with open(LOG_FILE, 'a', newline='') as f:
                w = csv.writer(f)
                w.writerow([epoch+1,
                            f"{train_loss:.4f}",
                            f"{train_loss_ce:.4f}",
                            f"{train_loss_l2:.4f}",
                            f"{val_loss:.4f}",
                            f"{val_loss_l2:.2f}",
                            f"{scheduler.get_last_lr()[0]:.2e}"])

            print(f"    Validation total loss: {val_loss:.4f} | Best: {best_val_loss:.4f}")
            print(f"    Validation L2: {val_loss_l2:.2f} | Best: {best_val_loss_l2:.2f}")

            # Save best model
            if val_loss_l2 > 0 and val_loss_l2 < best_val_loss_l2:
                
                improvement = (best_val_loss - val_loss) / best_val_loss * 100 if best_val_loss != float('inf') else 0.0
                improvement_l2 = (best_val_loss_l2 - val_loss_l2) / best_val_loss_l2 * 100 if best_val_loss_l2 != float('inf') else 0.0
                best_val_loss = val_loss
                best_val_loss_l2 = val_loss_l2

                # torch.save(model.state_dict(), LOG_DIR / "best.pth")
                torch.save({
                    'epoch': epoch + 1,
                    'model_state_dict': model.state_dict() if not isinstance(model, torch.nn.DataParallel) else model.module.state_dict(),
                    'val_loss': best_val_loss,
                    'val_loss_l2': best_val_loss_l2,
                    'optimizer_state_dict': optimizer.state_dict(),
                }, LOG_DIR / "best.pth")

                print(f"    NEW BEST! → Loss: {best_val_loss:.4f} (-{improvement:.2f}%) | L2: {best_val_loss_l2:.3f} (-{improvement_l2:.2f}%)")

    torch.save(model.state_dict(), LOG_DIR / "final.pth")
    torch.save({
        'epoch': NUM_EPOCHS,
        'model_state_dict': model.state_dict() if not isinstance(model, torch.nn.DataParallel) else model.module.state_dict(),
        'val_loss': val_loss,
        'val_loss_l2': val_loss_l2,
    }, LOG_DIR / "final.pth")
    print(f"\nTraining finished! Best L2: {best_val_loss:.2f}")

if __name__ == "__main__":
    main()
    # test_get_raw_gaze_point()

