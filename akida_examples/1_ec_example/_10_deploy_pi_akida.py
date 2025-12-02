
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent / "training"))
from training._3_4_train_fast_akida import get_out_gaze_point, AkidaGazeDataset
import numpy as np
import time
from datetime import datetime
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

from akida import Model
AKIDA_FOLDER_PATH = Path("akida_examples/1_ec_example/quantized_models/q8_calib_b8_n10/akida_models")
AKIDA_PATH = AKIDA_FOLDER_PATH / "akida_int8_v2.fbz"
akida_model = Model(str(AKIDA_PATH)) # important! load model from akida
akida_model = Model(str(AKIDA_PATH)) # important! load model from akida
print("\nLoaded Akida SNN model")
# akida_model.summary()






# ============================================================
# MAP MODEL TO REAL AKIDA HARDWARE
# ============================================================

from akida import devices 

# Detect real hardware
devices = akida.devices()
print(f'Available devices: {[dev.desc for dev in devices]}')
assert len(devices), "No device found, this example needs an Akida NSoC_v2 device."
device = devices[0]
assert device.version == akida.NSoC_v2, "Wrong device found, this example needs an Akida NSoC_v2."
print(f"Found Akida device: {device}")
print(f"Akida device IP version: {device.ip_version}\n") # IpVersion.v1 !!!

# Akida model properties
print(f"Akida model device: {akida_model.device}")
print(f"Akida model IP version: {akida_model.ip_version}") # IpVersion.v2
print(f"Akida model MACs: {akida_model.macs}\n")

# Akida model version
print(f"Akida module version: {akida.__version__}")

# Create v2 virtual device (emulates SixNodesIPv2)
device_v2 = akida.SixNodesIPv2()
print(f"Virtual device IP: {device_v2.ip_version}")  # IpVersion.v2


# Enable power measurement
# device.soc.power_measurement_enabled = True





# ============================================================
# RUN INFERENCE ON REAL AKIDA HARDWARE
# ============================================================

DATA_ROOT_AKIDA = Path("/home/pi/Jon/IndustrialProject/akida_examples/1_ec_example/training/preprocessed_akida_test_small")
test_dataset = AkidaGazeDataset(split="test", data_root=DATA_ROOT_AKIDA)
test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=4)

total_error_px = 0.0
total_samples = 0
total_model_time_ms = 0.0
total_time_ms = 0.0
# total_power_mw = 0.0

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
    pred = akida_model.predict(x)          # [B, H=3, W=4, C=3]
    latency_ms = (time.time() - start) * 1000

    # Floor power (mW)
    # floor_power = device_v2.soc.power_meter.floor

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
    # total_power_mw += floor_power

    pbar.set_postfix({"L2": total_error_px / total_samples ,
                      "Model Lat(ms)": f"{total_model_time_ms / total_samples:.2f}",
                      "Total Lat(ms)": f"{total_time_ms / total_samples:.2f}"})
    # "Total Power(mW)": f"{total_power_mw / total_samples:.2f}


# formatter for file sizes
def format_size(b):
    for unit in ['B', 'KB', 'MB', 'GB']:
        if b < 1024: return f"{b:.1f}{unit}"
        b /= 1024
    return f"{b:.1f}GB"

# Final result
akida_l2 = total_error_px / total_samples
akida_size = format_size(os.path.getsize(AKIDA_PATH)) if AKIDA_PATH.exists() else "N/A"
akida_latency_ms = total_model_time_ms / total_samples
akida_total_latency_ms = total_time_ms / total_samples
# akida_power_mw = total_power_mw / total_samples

print(f"\nAKIDA SNN FINAL RESULTS")
print(f"Total samples: {total_samples}")
print(f"Model size: {akida_size}")
print(f"Average L2 error: {akida_l2:.2f} px")
print(f"Average Model latency (ms): {akida_latency_ms:.2f} ms → {1000/akida_latency_ms:.0f} FPS")
print(f"Average Total latency (ms): {akida_total_latency_ms:.2f} ms → {1000/akida_total_latency_ms:.0f} FPS")
# print(f"Average Inference power (mW): {akida_power_mw:.2f} mW")









# ============================================================
# Save clean report 
# ============================================================
summary_path = AKIDA_FOLDER_PATH / "akida_report.txt"
with open(summary_path, "w") as f:
    f.write("Inference in Akida Chip on Raspberry Pi - Report\n")
    f.write("="*65 + "\n")
    f.write(f"{'Model':<12} {'Gaze L2 (px)':>15} {'Size':>15} {'Model Lat(ms)':>15} {'Total Lat(ms)':>15} \n")
    f.write("-"*65 + "\n")
    
    f.write(f"{'AKIDA_QINT8_PI':<12} {akida_l2:15.3f} px {akida_size:>15} {akida_latency_ms:>15.2f} {akida_total_latency_ms:>15.2f}\n")
    
    f.write("-"*65 + "\n")
    f.write(f"Test samples: {total_samples:,}\n")
    f.write(f"Original model: {AKIDA_PATH} ({akida_size})\n")
    f.write(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")

print(f"Clean L2 report saved → {summary_path}")