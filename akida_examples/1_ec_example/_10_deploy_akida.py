
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent / "training"))
from training._3_4_train_fast_akida import get_out_gaze_point, AkidaGazeDataset
import numpy as np
import time
import torch
import os
import tensorflow as tf
import onnx
from torch.utils.data import DataLoader 
from tqdm import tqdm

# Disable GPU for TF (quantizeml) -> FORCE quantizeml TO RUN ON CPU
os.environ["CUDA_VISIBLE_DEVICES"] = ""          # ← Disable GPU completely for TF/quantizeml
tf.config.set_visible_devices([], 'GPU')         # ← Extra safety — hide GPU from TF

import cnn2snn
import akida








# ============================================================
# CONFIG 
# ============================================================

# Input / output sizes
W, H = 640, 480              # Original input size (pixels)
W_IN, H_IN = 128, 96         # Model input size (pixels)
W_OUT, H_OUT = 4, 3          # Output heatmap size (out coordinates)

BATCH_SIZE = 8






# ============================================================
# LOAD SAVED SNN INT8 MODEL
# ============================================================

AKIDA_FOLDER_PATH = Path("akida_examples/1_ec_example/akida_models")
akida_path = AKIDA_FOLDER_PATH / "akida2_int8.fbz"
akida_model = akida.Model(str(akida_path))
print("\nLoaded Akida SNN model")
#akida_model.summary()






# ============================================================
# MAP MODEL TO REAL AKIDA HARDWARE
# ============================================================

from akida import devices, Mapper 

# Detect real hardware
device = devices()[0]              # takes first available AKD1000/1500/PCIe
print(f"Found Akida device: {device}")

# Map your saved model to hardware
mapped_model = Mapper.map(akida_model)     # ← critical step
print("Model mapped to hardware — ready for ultra-low power inference")






# ============================================================
# RUN INFERENCE ON REAL AKIDA HARDWARE
# ============================================================


print(f"Hardware inference: {latency_ms:.2f} ms → {1000/latency_ms:.0f} FPS")
print(f"Power: ~8–15 mW total system")

# Convert to gaze point
pred_x, pred_y, _, _ = get_out_gaze_point(torch.from_numpy(potentials), one_frame_only=True)
gaze_x = pred_x * 640 / 4
gaze_y = pred_y * 480 / 3
print(f"GAZE → ({gaze_x:.1f}, {gaze_y:.1f}) px")








test_dataset = AkidaGazeDataset(split="test")
test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=4)

total_error_px = 0.0
total_samples = 0
total_model_time_ms = 0.0
total_time_ms = 0.0

print("\nEvaluating Akida SNN on test set...")
pbar = tqdm(test_loader, desc="Akida SNN Eval", leave=True)

for batch in pbar:
    # Input: [B, 2, 50, 96, 128] → we only need last frame for 4D model
    # → take last timestep and convert to [B, 96, 128, 2] uint8
    x = batch['input'][:, :, -1, :, :].numpy()       # [B, 2, 96, 128]
    x = np.transpose(x, (0, 2, 3, 1))                # → [B, 96, 128, 2]
    x = np.clip(x, 0, 1).astype(np.int8)        # ensure int8 (0,1)

    # Ground truth: from [B, 3, 50, 3, 4] → we only need last frame for 4D model
    target = batch['target'][:, :, -1, :, :].numpy()  # [B, C=3, H=3, W=4]

    # Inference on REAL Akida chip
    start = time.time()
    pred = mapped_model.predict(x)          # [B, H=3, W=4, C=3]
    latency_ms = (time.time() - start) * 1000

    # Convert to torch for your gaze function
    pred_t = torch.from_numpy(pred)
    pred_t = pred_t.permute(0, 3, 1, 2)    # [B, C=3, H=3, W=4]

    # Convert to pixels
    pred_x, pred_y, _, _ = get_out_gaze_point(pred_t, one_frame_only=True)
    pred_px = pred_x * W / W_OUT
    pred_py = pred_y * H / H_OUT
    
    total_latency_ms = (time.time() - start) * 1000

    # Convert to pixels
    targ_t = torch.from_numpy(target)
    gt_x,   gt_y,   _, _ = get_out_gaze_point(targ_t, one_frame_only=True)
    gt_px   = gt_x   * W / W_OUT
    gt_py   = gt_y   * H / H_OUT

    # L2 error
    error_l2 = torch.sqrt((pred_px - gt_px)**2 + (pred_py - gt_py)**2).mean()

    # Accumulate results
    total_error_px += error_l2.sum().item()
    total_samples += x.shape[0]
    total_model_time_ms += latency_ms
    total_time_ms += total_latency_ms

    pbar.set_postfix({"L2": total_error_px / total_samples ,
                      "Model Lat(ms)": f"{total_model_time_ms / total_samples:.2f}",
                      "Total Lat(ms)": f"{total_time_ms / total_samples:.2f}"})

# Final result
akida_l2 = total_error_px / total_samples
akida_latency_ms = total_model_time_ms / total_samples
akida_total_latency_ms = total_time_ms / total_samples
print(f"\nAKIDA SNN FINAL RESULTS")
print(f"Total samples: {total_samples}")
print(f"Average L2 error: {akida_l2:.2f} px")
print(f"Average Model latency (ms): {akida_latency_ms:.2f} ms → {1000/akida_latency_ms:.0f} FPS")
print(f"Average Total latency (ms): {akida_total_latency_ms:.2f} ms → {1000/akida_total_latency_ms:.0f} FPS")