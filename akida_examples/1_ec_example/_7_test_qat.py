# _7_test_qat.py
# Evaluate QAT-quantized EyeTennSt (int8 or int4) on the test set
# → Reports: Gaze MAE (px), State Accuracy (%), total samples
# → Identical structure to 4_test_tennst.py — just loads quantized weights + converts model

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from pathlib import Path
import numpy as np
from tqdm import tqdm
from training._2_3_model_f16_last_10_gradual import EyeTennSt   # ← your final model with gradual head
from training._3_3_train_fast_dataset_10 import EyeTrackingDataset  # ← fast dataset

# Performance settings (same as training)
torch.backends.cuda.matmul.allow_tf32 = False
torch.backends.cudnn.benchmark = False
import os
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

# ============================================================
# CONFIG — UPDATE THESE PATHS
# ============================================================
DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# Point to your dataset
DATA_ROOT = Path("/home/dronelab-pc-1/Jon/IndustrialProject/akida_examples/1_ec_example/training/preprocessed_fast")

# Choose which QAT model to test
BITS = 8
QAT_DIR = Path(f"quantized_models/qat_int{BITS}")
MODEL_PATH = QAT_DIR / f"best_qat_int{BITS}.pth"   # ← or "final_int{BITS}_qat.pth"

BATCH_SIZE = 64
print(f"Testing QAT {BITS}-bit model on {DEVICE}")
print(f"Model: {MODEL_PATH}")
print(f"Dataset: {DATA_ROOT}")

# ============================================================
# MAIN TEST LOOP — SAME AS 4_test_tennst.py
# ============================================================
@torch.no_grad()
def main():
    test_dataset = EyeTrackingDataset(split="test")
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False,
                             num_workers=8, pin_memory=True)

    # Load base model (float16)
    torch.cuda.empty_cache()
    model = EyeTennSt(t_kernel_size=5, s_kernel_size=3, n_depthwise_layers=6).to(DEVICE)

    # Multi-GPU support
    if torch.cuda.device_count() > 1:
        print(f"Using {torch.cuda.device_count()} GPUs with DataParallel...")
        model = torch.nn.DataParallel(model)

    # TORCH.COMPILE — still great, now even more stable
    print("Compiling model with torch.compile()...")
    model = torch.compile(model, mode="reduce-overhead")

    # Load QAT quantized weights
    print(f"Loading QAT {BITS}-bit weights from: {MODEL_PATH}")
    state_dict = torch.load(MODEL_PATH, map_location=DEVICE)

    # CRITICAL: Convert model to quantized version BEFORE loading weights
    if BITS == 8:
        model.qconfig = torch.ao.quantization.get_default_qconfig('fbgemm')
    else:
        model.qconfig = torch.ao.quantization.get_default_qat_qconfig('fbgemm', version=4)

    # Prepare → fuse → convert (same flow as QAT training)
    model_fp32 = torch.ao.quantization.prepare(model.train())
    model_int = torch.ao.quantization.convert(model_fp32.eval())

    # Load the quantized state dict
    model_int.load_state_dict(state_dict)
    model_int.eval()
    model = model_int.to(DEVICE)

    print("Quantized QAT model loaded and ready")

    # Metrics
    total_gaze_error = 0.0
    total_state_correct = 0
    total_samples = 0

    print("Starting inference on quantized model...")
    for batch in tqdm(test_loader, desc=f"Testing QAT {BITS}-bit"):
        x = batch['input'].to(DEVICE, non_blocking=True)
        gaze_target = batch['gaze'].to(DEVICE)
        state_target = batch['state'].to(DEVICE)

        # with torch.cuda.amp.autocast():
        with torch.amp.autocast('cuda'):
            gaze_pred, state_logit = model(x)

        # Gaze MAE
        gaze_error = torch.mean(torch.abs(gaze_pred - gaze_target), dim=1)
        total_gaze_error += gaze_error.sum().item()

        # State accuracy
        state_pred = (torch.sigmoid(state_logit) > 0.5).float()
        total_state_correct += (state_pred == state_target).sum().item()

        total_samples += x.size(0)

    # Final results
    avg_gaze_mae = total_gaze_error / total_samples
    state_accuracy = total_state_correct / total_samples * 100

    print("\n" + "="*60)
    print(f"FINAL QAT {BITS}-BIT TEST RESULTS")
    print("="*60)
    print(f"Model: {MODEL_PATH.name}")
    print(f"Total test samples: {total_samples:,}")
    print(f"Gaze MAE:  {avg_gaze_mae:.3f} pixels")
    print(f"Blink Acc: {state_accuracy:.2f}%")
    print("="*60)

    # Save results
    result_file = QAT_DIR / "qat_test_results.txt"
    with open(result_file, 'w') as f:
        f.write(f"QAT {BITS}-bit Test Results\n")
        f.write(f"Model: {MODEL_PATH.name}\n")
        f.write(f"Gaze MAE: {avg_gaze_mae:.3f} px\n")
        f.write(f"Blink Accuracy: {state_accuracy:.2f}%\n")
        f.write(f"Total samples: {total_samples}\n")
    print(f"Results saved to: {result_file}")

if __name__ == "__main__":
    main()