# _6_train_quantized_pytorch.py
# Pure PyTorch Quantization-Aware Training (QAT) for int8 or int4
# → Uses torch.ao.quantization (official PyTorch) → perfect for Akida
# → Recovers nearly all accuracy lost in post-training quantization

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from pathlib import Path
import argparse
from tqdm import tqdm
import os
import signal

# TF32 can spike power on Ampere/Turing cards
torch.backends.cuda.matmul.allow_tf32 = False

# Benchmark mode can cause huge power spikes at start
torch.backends.cudnn.benchmark = False

import os
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

# ============================================================
# CONFIG
# ============================================================
DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
DATA_ROOT = Path("/home/dronelab-pc-1/Jon/IndustrialProject/akida_examples/1_ec_example/training/preprocessed_fast")
BATCH_SIZE = 64
NUM_EPOCHS = 8
LEARNING_RATE = 1e-4  # ← very small for QAT
WEIGHT_DECAY = 1e-5

# Argument: int4 or int8
BITS = 8

# Paths
QAT_DIR = Path(f"quantized_models/qat_int{BITS}")
QAT_DIR.mkdir(parents=True, exist_ok=True)
Q_PT_MODEL_PATH = f"quantized_models/q_tennst_int{BITS}.pt"   # ← you will save this from step 5 first

print(f"PyTorch QAT fine-tuning of {BITS}-bit model")
print(f"Loading base quantized model from: {Q_PT_MODEL_PATH}")

# ============================================================
# 1. Load your float16 model first
# ============================================================
torch.cuda.empty_cache()
model = EyeTennSt(t_kernel_size=5, s_kernel_size=3, n_depthwise_layers=6).to(DEVICE)

# DATA PARALLEL for multi-GPU setups
if torch.cuda.device_count() > 1:
    print(f"Using {torch.cuda.device_count()} GPUs with DataParallel...")
    model = torch.nn.DataParallel(model)

# TORCH.COMPILE — still great, now even more stable
print("Compiling model with torch.compile()...")
model = torch.compile(model, mode="reduce-overhead")

model.load_state_dict(torch.load("training/runs/your_best_model/best.pth", map_location=DEVICE))
model.eval()
print("Model loaded and set to eval mode")

# ============================================================
# 2. Apply Post-Training Quantization (PTQ) → int8 or int4
# ============================================================
if BITS == 8:
    model.qconfig = torch.ao.quantization.get_default_qconfig('fbgemm')
else:
    model.qconfig = torch.ao.quantization.get_default_qat_qconfig('fbgemm', version=4)  # int4

torch.ao.quantization.fuse_modules(model, [['conv', 'bn', 'relu']], inplace=True)  # if you have any
model = torch.ao.quantization.prepare_qat(model.train())  # ← QAT mode

# Load your PTQ weights (you save this after calibration in step 5)
if Path(Q_PT_MODEL_PATH).exists():
    model.load_state_dict(torch.load(Q_PT_MODEL_PATH, map_location=DEVICE))
    print("Loaded PTQ weights for QAT start")
else:
    print("Warning: PTQ model not found → starting from float16 (will be slower)")

# ============================================================
# 3. Dataset & loader
# ============================================================
train_dataset = EyeTrackingDataset(split="train")
val_dataset = EyeTrackingDataset(split="test")

train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True,
                          num_workers=12, pin_memory=True, persistent_workers=True)
val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False,
                        num_workers=12, pin_memory=True)

# ============================================================
# 4. Loss & optimizer
# ============================================================
criterion_gaze = nn.MSELoss()
criterion_state = nn.BCEWithLogitsLoss()
optimizer = optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
scaler = torch.cuda.amp.GradScaler()

# ============================================================
# 5. Training loop with QAT
# ============================================================
best_mae = float('inf')

for epoch in range(NUM_EPOCHS):
    model.train()
    train_loss = train_mae = 0.0
    pbar = tqdm(train_loader, desc=f"QAT Epoch {epoch+1}/{NUM_EPOCHS}")

    for batch in pbar:
        x = batch['input'].to(DEVICE, non_blocking=True)
        gaze_target = batch['gaze'].to(DEVICE, non_blocking=True)
        state_target = batch['state'].to(DEVICE, non_blocking=True)

        optimizer.zero_grad()

        # MIXED PRECISION FORWARD + BACKWARD
        # with torch.cuda.amp.autocast():
        with torch.amp.autocast('cuda'):
            gaze_pred, state_logit = model(x)
            loss_gaze = criterion_gaze(gaze_pred, gaze_target)
            loss_state = criterion_state(state_logit.squeeze(1), state_target)
            loss = 100.0 * loss_gaze + loss_state

        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        train_loss += loss.item()
        train_mae += torch.mean(torch.abs(gaze_pred - gaze_target)).item()

        pbar.set_postfix({'loss': f"{loss.item():.4f}", 'mae': f"{train_mae/(pbar.n+1):.3f}"})

    # Validation
    model.eval()
    val_mae = 0.0
    with torch.no_grad():
        for batch in val_loader:
            x = batch['input'].to(DEVICE, non_blocking=True)
            gaze_target = batch['gaze'].to(DEVICE)
            gaze_pred, _ = model(x)
            val_mae += torch.mean(torch.abs(gaze_pred - gaze_target)).item()
    val_mae /= len(val_loader)
    print(f"Validation Gaze MAE: {val_mae:.3f} px")

    # Save best
    if val_mae < best_mae:
        best_mae = val_mae
        torch.save(model.state_dict(), QAT_DIR / "best_qat_int{BITS}.pth")
        print("New best QAT model saved!")

# ============================================================
# 6. Convert to quantized model (final step)
# ============================================================
model.to('cpu')
model = torch.ao.quantization.convert(model.eval())
final_path = QAT_DIR / f"final_qat_int{BITS}.pth"
torch.save(model.state_dict(), final_path)
print(f"\nQAT complete! Final quantized model: {final_path}")