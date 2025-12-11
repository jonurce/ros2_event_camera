

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
import time
from tqdm import tqdm
import os
from datetime import datetime
import tensorflow as tf
import onnx
from onnx import shape_inference
from tenns_modules import export_to_onnx
import time

# Model and dataset
from training._2_5_model_uint8_akida import EyeTennSt
from training._3_4_train_fast_akida import AkidaGazeDataset, get_out_gaze_point 

# Performance settings
torch.backends.cuda.matmul.allow_tf32 = False
torch.backends.cudnn.benchmark = False
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"





# ============================================================
# CONFIG 
# ============================================================

# Paths
# MODEL_PATH = Path("/home/dronelab-pc-1/Jon/IndustrialProject/akida_examples/1_ec_example/training/runs/tennst_11_akida_b128_e100/best.pth") 
MODEL_PATH = Path("/home/dronelab-pc-1/Jon/IndustrialProject/akida_examples/1_ec_example/training/runs/tennst_16_akida_b128_e150_ce_12mse_origpx/best.pth") 

# Input / output sizes
W, H = 640, 480              # Original input size (pixels)
W_IN, H_IN = 128, 96         # Model input size (pixels)
W_OUT, H_OUT = 4, 3          # Output heatmap size (out coordinates)

BATCH_S = 8
MAX_BATCH_NUMBER = 10
MAX_TEST_BATCH = 200

# Output
QUANTIZED_FOLDER_PATH = Path(f"akida_examples/1_ec_example/quantized_models/q8_default_GOOD_b{BATCH_S}_n{MAX_BATCH_NUMBER}")
QUANTIZED_FOLDER_PATH.mkdir(exist_ok=True)

print(f"Loading model from: {MODEL_PATH}")

timer = 0.0
start_time = time.time()






# ============================================================
# 0. Load trained PyTorch model in GPU for evaluation
# ============================================================

# DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
DEVICE = "cpu"

# torch.cuda.empty_cache()
model = EyeTennSt(t_kernel_size=5, s_kernel_size=3, n_depthwise_layers=4).to(DEVICE)

checkpoint = torch.load(MODEL_PATH, map_location=DEVICE)

# Extract the actual state_dict (handles all saving formats)
if "model_state_dict" in checkpoint:
    state_dict = checkpoint["model_state_dict"]
    print(f"Loaded checkpoint from epoch {checkpoint.get('epoch', '?')}")
    print(f"   → val_loss_l2: {checkpoint.get('val_loss_l2', 'N/A'):.3f}")
elif "state_dict" in checkpoint:
    state_dict = checkpoint["state_dict"]
else:
    state_dict = checkpoint

# Load into model (handles DataParallel correctly)
if isinstance(model, torch.nn.DataParallel):
    model.module.state_dict(state_dict)
else:
    model.state_dict(state_dict)

model.eval()
print(f"PyTorch model loaded on {DEVICE} for evaluation")







# ============================================================
# 1. Evaluation on test set for Tennst before disabling GPU (using out cell + offset → real px)
# ============================================================

print("\n" + "="*70)
print("EVALUATING TENNST MODEL (Gaze L2 in original pixels)")
print("="*70)

test_dataset = AkidaGazeDataset(split="test")
test_loader = DataLoader(test_dataset, batch_size=BATCH_S, shuffle=False, num_workers=8, pin_memory=True)

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
print("\nEvaluating original tennst PyTorch model...")
model.to(DEVICE)
model.eval()

tennst_l2 = 0.0
samples = 0
tennst_model_time_ms = 0.0
tennst_time_ms = 0.0

with torch.no_grad():
    for i, batch in enumerate(tqdm(test_loader, total=MAX_TEST_BATCH)):
        if i >= MAX_TEST_BATCH:
            break
        
        x = batch['input'].to(DEVICE)      # [B,2,50,96,128] uint8
        y = batch['target'].to(DEVICE)     # [B,3,50,3,4] float32

        # Inference
        start = time.time()
        # with torch.amp.autocast('cuda'):
        #    pred = model(x)
        pred = model(x)
        latency_ms = (time.time() - start) * 1000

        pred_x, pred_y, _, _ = get_out_gaze_point(pred)
        pred_px = pred_x * W / W_OUT
        pred_py = pred_y * H / H_OUT
        total_latency_ms = (time.time() - start) * 1000

        gt_x,   gt_y,   _, _ = get_out_gaze_point(y)
        gt_px   = gt_x   * W / W_OUT
        gt_py   = gt_y   * H / H_OUT

        # L2 error
        error_l2 = torch.sqrt((pred_px - gt_px)**2 + (pred_py - gt_py)**2).mean()

        # Accumulate results
        tennst_l2 += error_l2.item()
        samples += x.size(0)
        tennst_model_time_ms += latency_ms
        tennst_time_ms += total_latency_ms

tennst_l2 /= samples
tennst_size_str = get_model_size(MODEL_PATH)
tennst_model_time_ms /= samples
tennst_time_ms /= samples

print("\nTennst PyTorch model evaluation complete:")
print(f"Tennst PyTorch model → Gaze L2: {tennst_l2:.3f} px | Size: {tennst_size_str} | Model Latency: {tennst_model_time_ms:.2f} ms | Total Latency: {tennst_time_ms:.2f} ms")

timer = time.time() - start_time
print(f"Time taken for Tennst evaluation: {timer:.2f} seconds")







# ============================================================
# 2. Load trained PyTorch model in CPU for quantization
# ============================================================

# Disable GPU for TF (quantizeml) -> FORCE quantizeml TO RUN ON CPU
os.environ["CUDA_VISIBLE_DEVICES"] = ""          # ← Disable GPU completely for TF/quantizeml
tf.config.set_visible_devices([], 'GPU')         # ← Extra safety — hide GPU from TF

# Akida quantization tools
from quantizeml.models import quantize, QuantizationParams
from quantizeml import save_model

torch.cuda.empty_cache()
model = EyeTennSt(t_kernel_size=5, s_kernel_size=3, n_depthwise_layers=4).to("cpu")

checkpoint = torch.load(MODEL_PATH, map_location="cpu")

# Extract the actual state_dict (handles all saving formats)
if "model_state_dict" in checkpoint:
    state_dict = checkpoint["model_state_dict"]
    print(f"Loaded checkpoint from epoch {checkpoint.get('epoch', '?')}")
    print(f"   → val_loss_l2: {checkpoint.get('val_loss_l2', 'N/A'):.3f}")
elif "state_dict" in checkpoint:
    state_dict = checkpoint["state_dict"]
else:
    state_dict = checkpoint

# Load into model (handles DataParallel correctly)
if isinstance(model, torch.nn.DataParallel):
    model.module.state_dict(state_dict)
else:
    model.state_dict(state_dict)

model.eval().cpu()
print("PyTorch model loaded on CPU for quantization")








# ============================================================
# 3. Export to ONNX (quantizeml accepts ONNX directly) — MUST BE 4D: (B, C* T, H, W) for quantization
# ============================================================

print("Exporting to ONNX...")

# Using a batch size
onnx_path = QUANTIZED_FOLDER_PATH / "tennst.onnx"
export_to_onnx(model, (BATCH_S, 2, 50, 96, 128), out_path=str(onnx_path))

print(f"ONNX exported → {onnx_path}")

model_onnx = onnx.load(str(onnx_path))
print("ONNX model loaded")
print("Input shape:", [x.dim_value or x.dim_param for x in model_onnx.graph.input[0].type.tensor_type.shape.dim])
print("Output shape:", [x.dim_value or x.dim_param for x in model_onnx.graph.output[0].type.tensor_type.shape.dim])







# ============================================================
# 5. QUANTIZATION
# ============================================================


print("\nQuantizing to 8-bit...")

qparams_8bit = QuantizationParams(
    input_dtype='int8',
    input_weight_bits=8,
    weight_bits=8, # could be 4 for Akida 1.0
    activation_bits=8, # could be 4 for Akida 1.0
    # per_tensor_activations=True
    # output_bits=8,
    # buffer_bits=32 (default)
)

# per_tensor_activations (default False -> per-axis): defines quantization for ReLU activations.
# Per-axis = more accurate results, but more challenging to calibrate.
# Akida 1.0 only supports per-tensor activations

model_q8 = quantize(
    model_onnx,
    qparams=qparams_8bit,
    # samples=calibration_samples, # without real samples, it will use [-127, 128] values
    # num_samples = max_batch_number * BATCH_S, # default = 1024
    # batch_size=BATCH_S, default = 100
    # epochs=1,
)

print("8-bit quantization successful!")








# ============================================================
# 6. Save quantized model
# ============================================================

INT8_PATH = QUANTIZED_FOLDER_PATH / "q_tennst_int8.onnx"
save_model(model_q8, INT8_PATH)
print(f"\nQuantized model saved:")
print(f"  INT8 → {INT8_PATH}")

print("\nVerifying saved INT8 model structure...")
model_q8_onnx_int8 = onnx.load(str(INT8_PATH))
print("Int8 ONNX model loaded")
print("Input shape:", [x.dim_value or x.dim_param for x in model_q8_onnx_int8.graph.input[0].type.tensor_type.shape.dim])
print("Output shape:", [x.dim_value or x.dim_param for x in model_q8_onnx_int8.graph.output[0].type.tensor_type.shape.dim])


timer = time.time() - start_time
print(f"Total time taken for quantization process: {timer:.2f} seconds")







# ============================================================
# 7. Evaluation on test set (using out cell + offset → real px)
# ============================================================

from onnxruntime import InferenceSession, SessionOptions
from onnxruntime_extensions import get_library_path
from quantizeml.onnx_support.quantization import ONNXModel
from quantizeml.models import reset_buffers

print("\n" + "="*70)
print("EVALUATING QUANTIZED MODEL (Gaze L2 in original pixels)")
print("="*70)

# Evaluate quantized model

# formatter for file sizes
def format_size(b):
    for unit in ['B', 'KB', 'MB', 'GB']:
        if b < 1024: return f"{b:.1f}{unit}"
        b /= 1024
    return f"{b:.1f}GB"

# evaluate any ONNX model and return L2 error in pixels
def evaluate_model(model_q, model_path, name):
    sess_options = SessionOptions()
    sess_options.register_custom_ops_library(get_library_path())
    model_quant = ONNXModel(model_q)
    session = InferenceSession(
        model_quant.serialized,
        sess_options=sess_options,
        providers=['CPUExecutionProvider']
    )
    # session = InferenceSession(str(model_path))

    total_error_px = 0.0
    total_samples = 0
    total_q8_model_time_ms = 0.0
    total_q8_time_ms = 0.0
    MAX_TEST_BATCH = 20

    for i, batch in enumerate(tqdm(test_loader, total=MAX_TEST_BATCH)):
        if i >= MAX_TEST_BATCH:
            break
        
        latency_ms = 0.0
        total_latency_ms = 0.0

        # single batch inference (over frames)
        for frame_idx in range(batch['input'].shape[2]):
            # batch['input'] -> [B,2,50,96,128] uint8 
            x_np = batch['input'][:, :, frame_idx, ...].float().numpy()   # [B,2,96,128] float32

            # batch['target'] -> [B,3,50,3,4] float32
            target = batch['target'][:, :, frame_idx, ...].numpy()        # [B,3,3,4] float32

            # Inference
            start = time.time()
            pred = session.run(None, {model_quant.input[0].name: x_np})[0] # [B,3,3,4]
            latency_ms += (time.time() - start) * 1000

            # Convert to torch for get_out_gaze_point
            pred_t = torch.from_numpy(pred)
            
            # Extract gaze points in out cooridnates +  Convert to original pixels: *160
            pred_x, pred_y, _, _ = get_out_gaze_point(pred_t, one_frame_only=True)
            pred_px = pred_x * W / W_OUT 
            pred_py = pred_y * H / H_OUT

            total_latency_ms += (time.time() - start) * 1000

            # Extract gaze points in out cooridnates +  Convert to original pixels: *160
            targ_t = torch.from_numpy(target)
            gt_x,   gt_y,   _, _ = get_out_gaze_point(targ_t, one_frame_only=True)
            gt_px   = gt_x   * W / W_OUT
            gt_py   = gt_y   * H / H_OUT
            
        
        # measure only on last frame
        error_l2_px = torch.sqrt((pred_px - gt_px)**2 + (pred_py - gt_py)**2).mean()
        total_error_px += error_l2_px.item()
        total_samples += x_np.shape[0]
        total_q8_model_time_ms += latency_ms
        total_q8_time_ms += total_latency_ms

        # Reset FIFOs between each file
        reset_buffers(model_quant)

    avg_error_l2_px = total_error_px / (total_samples or 1)
    size_str = format_size(os.path.getsize(model_path)) if model_path.exists() else "N/A"
    q8_latency_ms = total_q8_model_time_ms / total_samples
    q8_total_latency_ms = total_q8_time_ms / total_samples
    return {"name": name, "l2": avg_error_l2_px, "size_str": size_str,
            "samples": total_samples, "model_latency_ms": q8_latency_ms,
            "total_latency_ms": q8_total_latency_ms}

print("\nEvaluating quantized INT8 model...")
int8_res = evaluate_model(model_q8, INT8_PATH, "INT8")
print(f"INT8 → {int8_res['l2']:.3f} px | Size: {int8_res['size_str']} | Model Latency: {int8_res['model_latency_ms']:.2f} ms | Total Latency: {int8_res['total_latency_ms']:.2f} ms")








# ============================================================
# 8. Final results table (L2 error in pixels)
# ============================================================
print("\n" + "="*85)
print("FINAL QUANTIZATION RESULTS — Gaze L2 Error (pixels) + Model Size")
print("="*85)
print(f"{'Model':<12} {'Gaze L2 (px)':>18} {'Δ(px) vs Tennst':>16} {'Size':>15} {'Reduction':>12} {'Model Lat(ms)':>15} {'Δ(%) vs Tennst':>16} {'Total Lat(ms)':>15} {'Δ(%) vs Tennst':>16}")
print("-"*85)

# Tennst baseline
print(f"{'Tennst':<12} {tennst_l2:18.3f} {'N/A':>16} {tennst_size_str:>15} {'N/A':>12} {tennst_model_time_ms:>15.2f} {'N/A':>16} {tennst_time_ms:>15.2f} {'N/A':>16}")

# INT8
delta_l2 = int8_res['l2'] - tennst_l2
reduction_8 = "N/A" if tennst_size_str == "N/A" or int8_res['size_str'] == "N/A" else \
    f"{(1 - os.path.getsize(INT8_PATH) / MODEL_PATH.stat().st_size)*100:5.1f}%"
delta_model_lat_8 = "N/A" if tennst_model_time_ms == 0 or int8_res['model_latency_ms'] == 0 else \
    ( int8_res['model_latency_ms'] - tennst_model_time_ms ) / tennst_model_time_ms
delta_total_lat_8 = "N/A" if tennst_time_ms == 0 or int8_res['total_latency_ms'] == 0 else \
    ( int8_res['total_latency_ms'] - tennst_time_ms ) / tennst_time_ms
print(f"{'Q_INT8':<12} {int8_res['l2']:18.3f} {delta_l2:+8.3f} px {int8_res['size_str']:>15} {reduction_8:>14} {int8_res['model_latency_ms']:>15.2f} {delta_model_lat_8:+15.2%} {int8_res['total_latency_ms']:>15.2f} {delta_total_lat_8:+15.2%}")

print("-"*85)
print(f"Total test samples: {samples:,}")
print(f"Original model: {MODEL_PATH}")
print(f"Quantized folder: {QUANTIZED_FOLDER_PATH}")
print("="*85)






# ============================================================
# 9. Save clean report (L2 only)
# ============================================================
summary_path = QUANTIZED_FOLDER_PATH / "quantization_report.txt"
with open(summary_path, "w") as f:
    f.write("Quantization Report\n")
    f.write("="*65 + "\n")
    f.write(f"{'Model':<12} {'Gaze L2 (px)':>15} {'Δ(px) vs Tennst':>15} {'Size':>15} {'Reduction':>12} {'Model Lat(ms)':>15} {'Δ(%) vs Tennst':>16} {'Total Lat(ms)':>15} {'Δ(%) vs Tennst':>16} \n")
    f.write("-"*65 + "\n")
    
    f.write(f"{'Tennst':<12} {tennst_l2:15.3f} {'N/A':>15} {tennst_size_str:>15} {'N/A':>12} {tennst_model_time_ms:>15.2f} {'N/A':>16} {tennst_time_ms:>15.2f} {'N/A':>16}\n")
    
    delta8 = int8_res['l2'] - tennst_l2
    red8 = "N/A" if tennst_size_str == "N/A" or int8_res['size_str'] == "N/A" else f"{(1 - os.path.getsize(INT8_PATH) / MODEL_PATH.stat().st_size)*100:5.1f}%"
    delta_model_lat_8 = "N/A" if tennst_model_time_ms == 0 or int8_res['model_latency_ms'] == 0 else \
        ( int8_res['model_latency_ms'] - tennst_model_time_ms ) / tennst_model_time_ms
    delta_total_lat_8 = "N/A" if tennst_time_ms == 0 or int8_res['total_latency_ms'] == 0 else \
        ( int8_res['total_latency_ms'] - tennst_time_ms ) / tennst_time_ms
    f.write(f"{'Q_INT8':<12} {int8_res['l2']:15.3f} {delta8:+8.3f} px {int8_res['size_str']:>15} {red8:>12} {int8_res['model_latency_ms']:>15.2f} {delta_model_lat_8:+15.2%} {int8_res['total_latency_ms']:>15.2f} {delta_total_lat_8:+15.2%}\n")
    
    f.write("-"*65 + "\n")
    f.write(f"Test samples: {samples:,}\n")
    f.write(f"Original model: {MODEL_PATH} ({tennst_size_str})\n")
    f.write(f"Quantized folder: {QUANTIZED_FOLDER_PATH}\n")
    f.write(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")

print(f"Clean L2 report saved → {summary_path}")

timer = time.time() - start_time
print(f"Total time taken for quantization evaluation: {timer:.2f} seconds")

