# _5_quantize_akida.py
# Quantize trained EyeTennSt (Akida-style) → int8 / int4 for Akida SNN conversion
# Now works with: uint8 [2,50,96,128] input + heatmap [3,50,3,4] output
# Uses quantizeml (official BrainChip tool)

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent / "training"))

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
import numpy as np
from tqdm import tqdm
import os
from datetime import datetime
import tensorflow as tf
import onnx
from onnx import shape_inference

# Disable GPU for TF (quantizeml) -> FORCE quantizeml TO RUN ON CPU
os.environ["CUDA_VISIBLE_DEVICES"] = ""          # ← Disable GPU completely for TF/quantizeml
tf.config.set_visible_devices([], 'GPU')         # ← Extra safety — hide GPU from TF

# Akida quantization tools
from quantizeml.models import quantize
from quantizeml import load_model
from quantizeml.layers import QuantizationParams

# Model and dataset
from training._2_5_model_uint8_akida import EyeTennSt
from training._3_4_train_fast_akida import AkidaGazeDataset, get_out_gaze_point 

# Performance settings
# torch.backends.cuda.matmul.allow_tf32 = False
# torch.backends.cudnn.benchmark = False
# os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

# ============================================================
# CONFIG 
# ============================================================
# DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
DEVICE = torch.device("cpu")

# Paths
MODEL_PATH = Path("/home/dronelab-pc-1/Jon/IndustrialProject/akida_examples/1_ec_example/training/runs/tennst_11_akida_b128_e100/best.pth") 
DATA_ROOT = Path("/home/dronelab-pc-1/Jon/IndustrialProject/akida_examples/1_ec_example/training/preprocessed_fast")

# Input / output sizes
W, H = 640, 480              # Original input size (pixels)
W_IN, H_IN = 128, 96         # Model input size (pixels)
W_OUT, H_OUT = 4, 3          # Output heatmap size (out coordinates)

# Output
QUANTIZED_FOLDER_PATH = Path("akida_examples/1_ec_example/quantized_models")
QUANTIZED_FOLDER_PATH.mkdir(exist_ok=True)

print(f"Loading model from: {MODEL_PATH}")
print(f"Using calibration data from: {DATA_ROOT/'train'}")

# ============================================================
# 1. Load trained PyTorch model
# ============================================================

torch.cuda.empty_cache()
model = EyeTennSt(t_kernel_size=5, s_kernel_size=3, n_depthwise_layers=4).to(DEVICE)

checkpoint = torch.load(MODEL_PATH, map_location="cpu")

# Extract the actual state_dict (handles all saving formats)
if "model_state_dict" in checkpoint:
    state_dict = checkpoint["model_state_dict"]
    print(f"Loaded checkpoint from epoch {checkpoint.get('epoch', '?')}")
    print(f"   → val_loss: {checkpoint.get('val_loss', 'N/A'):.4f} | val_loss_l2: {checkpoint.get('val_loss_l2', 'N/A'):.3f}")
elif "state_dict" in checkpoint:
    state_dict = checkpoint["state_dict"]
else:
    state_dict = checkpoint

# Load into model (handles DataParallel correctly)
if isinstance(model, torch.nn.DataParallel):
    model.module.state_dict(state_dict)
else:
    model.state_dict(state_dict)

model.float().eval()
print("PyTorch model loaded")

# ============================================================
# 2. Export to ONNX (quantizeml accepts ONNX directly) 
# ============================================================

print("Exporting to ONNX...")

# uint8 input connot be used for quantization -> use float32 dummy input scaled [0,1]
dummy_input = torch.randint(0, 256, (1, 2, 50, 96, 128), dtype=torch.float32, device="cpu") / 255.0  

onnx_path = QUANTIZED_FOLDER_PATH / "tennst.onnx"
torch.onnx.export(
    model,
    dummy_input,
    onnx_path,
    export_params=True,
    opset_version=17,
    do_constant_folding=True,
    input_names=["input"],
    output_names=["output"],
    dynamic_axes={"input": {0: "batch"}}
)
print(f"ONNX exported → {onnx_path}")

# Load as Keras model -> wrapped for quantizeml
model_keras = load_model(str(onnx_path))
print("Keras model loaded from ONNX")

# ============================================================
# 3. Calibration dataset (from preprocessed_akida)
# ============================================================

calib_dataset = AkidaGazeDataset(split="train")
calib_loader = DataLoader(calib_dataset, batch_size=64, shuffle=True, num_workers=8, pin_memory=True)

calibration_samples = []
print("Collecting calibration samples (float32 voxels)...")
for i, batch in enumerate(tqdm(calib_loader, total=25)):
    if i >= 25:
        break
    # x = batch['input'].numpy()    # ← [B,2,50,96,128] uint8 → numpy
    x = batch['input'].float().div(255.0).cpu().numpy()   # ← float32 + normalized to [0,1]
    calibration_samples.append(x)

calibration_samples = np.concatenate(calibration_samples, axis=0)
print(f"Collected {calibration_samples.shape[0]} calibration samples")

# ============================================================
# 4. QUANTIZATION
# ============================================================

print("\nQuantizing to 8-bit...")
qparams_8bit = QuantizationParams(
    input_weight_bits=8,
    weight_bits=8,
    activation_bits=8,
    per_tensor_activations=True
)

model_q8 = quantize(
    model_keras,
    qparams=qparams_8bit,
    samples=calibration_samples,
    num_samples=1024,
    batch_size=64,
    epochs=1
)

print("\nQuantizing to 4-bit...")
qparams_4bit = QuantizationParams(
    input_weight_bits=8,
    weight_bits=4,
    activation_bits=4,
    per_tensor_activations=True
)
model_q4 = quantize(
    model_keras,
    qparams=qparams_4bit,
    samples=calibration_samples,
    num_samples=1024,
    batch_size=64,
    epochs=1
)

# ============================================================
# 5. Save quantized models
# ============================================================
INT8_PATH = QUANTIZED_FOLDER_PATH / "q_tennst_int8.keras"
INT4_PATH = QUANTIZED_FOLDER_PATH / "q_tennst_int4.keras"

model_q8.save(INT8_PATH)
model_q4.save(INT4_PATH)

print(f"\nQuantized models saved:")
print(f"  INT8 → {INT8_PATH}")
print(f"  INT4 → {INT4_PATH}")

# ============================================================
# 6. Evaluation on test set (using out cell + offset → real px)
# ============================================================

print("\n" + "="*70)
print("EVALUATING QUANTIZED MODELS (Gaze L2 in original pixels)")
print("="*70)

test_dataset = AkidaGazeDataset(split="test")
test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False, num_workers=8, pin_memory=True)

# evaluate any Keras model and return L2 error in pixels
def evaluate_model(model_keras, name, path=None):
    total_error_px = 0.0
    total_samples = 0
    pbar = tqdm(test_loader, desc=f"Eval {name}", leave=False)
    for batch in pbar:
        # x_np = batch['input'].numpy()           # [B,2,50,96,128] uint8
        x = batch['input'].float() / 255.0             # [B, 2, 50, 96, 128]
        x_np = x.numpy()

        target = batch['target'].numpy()        # [B,3,50,3,4] float32

        # Keras inference
        pred = model_keras.predict(x_np, verbose=0)  # [B,3,50,3,4]

        # Convert to torch for get_out_gaze_point
        pred_t = torch.from_numpy(pred)
        targ_t = torch.from_numpy(target)

        # Extract gaze points in out cooridnates
        pred_x, pred_y, _, _ = get_out_gaze_point(pred_t)
        gt_x,   gt_y,   _, _ = get_out_gaze_point(targ_t)

        # Convert to original pixels: *160
        pred_px = pred_x * W / W_OUT 
        pred_py = pred_y * H / H_OUT
        gt_px   = gt_x   * W / W_OUT
        gt_py   = gt_y   * H / H_OUT

        error_l2_px = torch.sqrt((pred_px - gt_px)**2 + (pred_py - gt_py)**2).mean()
        total_error_px += error_l2_px.item()
        total_samples += x_np.shape[0]

    avg_error_l2_px = total_error_px / total_samples
    size_str = format_size(os.path.getsize(path)) if path and path.exists() else "N/A"
    return {"name": name, "l2": avg_error_l2_px, "size_str": size_str, "samples": total_samples}

# formatter for file sizes
def format_size(b):
    for unit in ['B', 'KB', 'MB', 'GB']:
        if b < 1024: return f"{b:.1f}{unit}"
        b /= 1024
    return f"{b:.1f}GB"

# get model size or N/A
def get_model_size(path):
    if not path.exists():
        return "N/A"
    bytes_size = path.stat().st_size
    for unit in ['B', 'KB', 'MB', 'GB']:
        if bytes_size < 1024:
            return f"{bytes_size:.1f}{unit}"
        bytes_size /= 1024
    return f"{bytes_size:.1f}GB"

# Evaluate tennst PyTorch model (baseline)
print("Evaluating original tennst PyTorch model...")
model.to(DEVICE)
model.eval()
tennst_l2 = 0.0
samples = 0
with torch.no_grad():
    for batch in tqdm(test_loader, desc="Tennst baseline", leave=False):
        x = batch['input'].to(DEVICE)
        y = batch['target'].to(DEVICE)
        with torch.amp.autocast('cuda'):
            pred = model(x)
        pred_x, pred_y, _, _ = get_out_gaze_point(pred)
        gt_x,   gt_y,   _, _ = get_out_gaze_point(y)
        error_l2 = torch.sqrt(( (pred_x - gt_x) * W / W_OUT )**2 + ( (pred_y - gt_y) * H / H_OUT )**2).mean()
        tennst_l2 += error_l2.item()
        samples += x.size(0)
tennst_l2 /= samples
tennst_size_str = get_model_size(MODEL_PATH)

# Evaluate quantized
int8_res = evaluate_model(model_q8, "INT8", INT8_PATH)
int4_res = evaluate_model(model_q4, "INT4", INT4_PATH)

# ============================================================
# 7. Final results table (L2 error in pixels)
# ============================================================
print("\n" + "="*85)
print("FINAL QUANTIZATION RESULTS — Gaze L2 Error (pixels) + Model Size")
print("="*85)
print(f"{'Model':<12} {'Gaze L2 (px)':>18} {'Δ vs Tennst':>16} {'Size':>15} {'Reduction':>12}")
print("-"*85)

# Tennst baseline
print(f"{'Tennst':<12} {tennst_l2:18.3f} {'-':>16} {tennst_size_str:>15} {'-':>12}")

# INT8
delta_l2 = int8_res['l2'] - tennst_l2
reduction_8 = "N/A" if tennst_size_str == "N/A" or int8_res['size_str'] == "N/A" else \
    f"{(1 - os.path.getsize(INT8_PATH) / MODEL_PATH.stat().st_size)*100:5.1f}%"
print(f"{'INT8':<12} {int8_res['l2']:18.3f} {delta_l2:+8.3f} px {int8_res['size_str']:>15} {reduction_8:>14}")

# INT4
delta_l2 = int4_res['l2'] - tennst_l2
reduction_4 = "N/A" if tennst_size_str == "N/A" or int4_res['size_str'] == "N/A" else \
    f"{(1 - os.path.getsize(INT4_PATH) / MODEL_PATH.stat().st_size)*100:5.1f}%"
print(f"{'INT4':<12} {int4_res['l2']:18.3f} {delta_l2:+8.3f} px {int4_res['size_str']:>15} {reduction_4:>14}")

print("-"*85)
print(f"Total test samples: {samples:,}")
print(f"Original model: {MODEL_PATH}")
print(f"Quantized folder: {QUANTIZED_FOLDER_PATH}")
print("="*85)

# ============================================================
# 8. Save clean report (L2 only)
# ============================================================
summary_path = QUANTIZED_FOLDER_PATH / "quantization_report.txt"
with open(summary_path, "w") as f:
    f.write("Akida-Style EyeTennSt Quantization Report\n")
    f.write("="*65 + "\n")
    f.write(f"{'Model':<12} {'Gaze L2 (px)':>15} {'Δ vs Tennst':>15} {'Size':>15} {'Reduction':>12}\n")
    f.write("-"*65 + "\n")
    
    f.write(f"{'Tennst':<12} {tennst_l2:15.3f} {'-':>15} {tennst_size_str:>15} {'-':>12}\n")
    
    delta8 = int8_res['l2'] - tennst_l2
    red8 = "N/A" if tennst_size_str == "N/A" else f"{(1 - os.path.getsize(INT8_PATH) / MODEL_PATH.stat().st_size)*100:5.1f}%"
    f.write(f"{'INT8':<12} {int8_res['l2']:15.3f} {delta8:+8.3f} px {int8_res['size_str']:>15} {red8:>12}\n")
    
    delta4 = int4_res['l2'] - tennst_l2
    red4 = "N/A" if tennst_size_str == "N/A" else f"{(1 - os.path.getsize(INT4_PATH) / MODEL_PATH.stat().st_size)*100:5.1f}%"
    f.write(f"{'INT4':<12} {int4_res['l2']:15.3f} {delta4:+8.3f} px {int4_res['size_str']:>15} {red4:>12}\n")
    
    f.write("-"*65 + "\n")
    f.write(f"Test samples: {samples:,}\n")
    f.write(f"Original model: {MODEL_PATH} ({tennst_size_str})\n")
    f.write(f"Quantized folder: {QUANTIZED_FOLDER_PATH}\n")
    f.write(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")

print(f"Clean L2 report saved → {summary_path}")

