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
import keras
from tf_keras.optimizers import Adam

# Model and dataset
from training._2_5_model_uint8_akida import EyeTennSt
from training._3_4_train_fast_akida import AkidaGazeDataset, get_out_gaze_point 

# Disable GPU for TF (quantizeml) -> FORCE quantizeml TO RUN ON CPU
os.environ["CUDA_VISIBLE_DEVICES"] = ""          # ← Disable GPU completely for TF/quantizeml
tf.config.set_visible_devices([], 'GPU')         # ← Extra safety — hide GPU from TF

# Akida quantization tools
import quantized_models
import quantizeml
from quantizeml.models import quantize, QuantizationParams
from quantizeml import save_model, load_model





# ============================================================
# CONFIG 
# ============================================================

# Input / output sizes
W, H = 640, 480              # Original input size (pixels)
W_IN, H_IN = 128, 96         # Model input size (pixels)
W_OUT, H_OUT = 4, 3          # Output heatmap size (out coordinates)

BATCH_S = 8
MAX_BATCH_NUMBER = 10

# Output
QUANTIZED_FOLDER_PATH = Path(f"akida_examples/1_ec_example/quantized_models/q8_calib_b{BATCH_S}_n{MAX_BATCH_NUMBER}")
QUANTIZED_FOLDER_PATH.mkdir(exist_ok=True)

QAT_FOLDER_PATH = QUANTIZED_FOLDER_PATH / "qat_models"
QAT_FOLDER_PATH.mkdir(exist_ok=True)

TENNST_ONNX_PATH = QUANTIZED_FOLDER_PATH / "tennst.onnx"
Q8_PATH = QAT_FOLDER_PATH / "q8_tennst.onnx"
QAT8_PATH = QAT_FOLDER_PATH / "qat_tennst.onnx"

# DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
DEVICE = "cpu"









# ============================================================
# Dataset for Calibration and Quantization Aware Training (QAT)
# ============================================================

train_dataset = AkidaGazeDataset(split="train")
train_loader = DataLoader(train_dataset, batch_size=BATCH_S, shuffle=True, num_workers=8, pin_memory=True)

x_train = []
y_train = []
samples = 0

print("Collecting calibration samples (float32 voxels)...")
for i, batch in enumerate(tqdm(train_loader, total=MAX_BATCH_NUMBER)):
    if i >= MAX_BATCH_NUMBER:
        break
    # batch['input'] -> [B,2,50,96,128] uint8 
    for frame_idx in range(batch['input'].shape[2]): # Iterate over time frames
        x = batch['input'][:, :, frame_idx, ...].float().numpy()  # [B, 2, 96, 128]
        y = batch['target'][:, :, frame_idx, ...].numpy()       # [B, 3, 3, 4]

        x_train.append(x)
        y_train.append(y)
        samples += x.shape[0]

    
# Should be = max_batch_number * BATCH_S
print("\nShould have collected:", MAX_BATCH_NUMBER * 50, "batches of size", BATCH_S)
print(f"Collected {len(x_train)} input batches, with shape {x_train[0].shape}")
print(f"Collected {len(y_train)} target batches, with shape {y_train[0].shape}")

x_train = np.array(x_train) 
y_train = np.array(y_train)
print(f"x_train shape: {x_train.shape}, y_train shape: {y_train.shape}")

x_train = np.concatenate(x_train, axis=0)
y_train = np.concatenate(y_train, axis=0)
print(f"Input shape after concatenate: {x_train.shape}")
print(f"Target shape after concatenate: {y_train.shape}")








# ============================================================
# QUANTIZATION
# ============================================================

print("\nLoading trained TennSt model for quantization...")
model_onnx = onnx.load(str(TENNST_ONNX_PATH))
print("ONNX model loaded")
print("Input shape:", [x.dim_value or x.dim_param for x in model_onnx.graph.input[0].type.tensor_type.shape.dim])
print("Output shape:", [x.dim_value or x.dim_param for x in model_onnx.graph.output[0].type.tensor_type.shape.dim])


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
    samples=x_train, # without real samples, it will use [-127, 128] values
    num_samples = MAX_BATCH_NUMBER * 50, # default = 1024
    batch_size=BATCH_S, # default = 100
    epochs=1,
)

print("8-bit quantization successful!")







# ============================================================
# SAVE QUANTIZED MODEL
# ============================================================

save_model(model_q8, Q8_PATH)
print(f"\nQuantized model saved:")
print(f"  Q8 → {Q8_PATH}")

print("\nVerifying saved Q8 model structure...")
model_q8_onnx_int8 = onnx.load(str(Q8_PATH))
print("Q8 ONNX model loaded")
print("Input shape:", [x.dim_value or x.dim_param for x in model_q8_onnx_int8.graph.input[0].type.tensor_type.shape.dim])
print("Output shape:", [x.dim_value or x.dim_param for x in model_q8_onnx_int8.graph.output[0].type.tensor_type.shape.dim])







# ============================================================
# EVALUATE Q8 IN TEST DATASET (REAL PX L2 ERROR)
# ============================================================

from onnxruntime import InferenceSession, SessionOptions
from onnxruntime_extensions import get_library_path
from quantizeml.onnx_support.quantization import ONNXModel
from quantizeml.models import reset_buffers

test_dataset = AkidaGazeDataset(split="test")
test_loader = DataLoader(test_dataset, batch_size=BATCH_S, shuffle=False, num_workers=8, pin_memory=True)


print("\n" + "="*70)
print("EVALUATING QUANTIZED MODEL (L2 IN REAL PIXELS)")
print("="*70)

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

    for i, batch in enumerate(tqdm(test_loader, total=MAX_BATCH_NUMBER)):
        if i >= MAX_BATCH_NUMBER:
            break
        for frame_idx in range(batch['input'].shape[2]):
            # batch['input'] -> [B,2,50,96,128] uint8 
            x_np = batch['input'][:, :, frame_idx, ...].float().numpy()   # [B,2,96,128] float32

            # batch['target'] -> [B,3,50,3,4] float32
            target = batch['target'][:, :, frame_idx, ...].numpy()        # [B,3,3,4] float32

            # Inference
            start = time.time()
            pred = session.run(None, {model_quant.input[0].name: x_np})[0] # [B,3,3,4]
            latency_ms = (time.time() - start) * 1000

            # Convert to torch for get_out_gaze_point
            pred_t = torch.from_numpy(pred)
            targ_t = torch.from_numpy(target)

            # Extract gaze points in out cooridnates
            pred_x, pred_y, _, _ = get_out_gaze_point(pred_t, one_frame_only=True)
            gt_x,   gt_y,   _, _ = get_out_gaze_point(targ_t, one_frame_only=True)

            # Convert to original pixels: *160
            pred_px = pred_x * W / W_OUT 
            pred_py = pred_y * H / H_OUT
            gt_px   = gt_x   * W / W_OUT
            gt_py   = gt_y   * H / H_OUT

            total_latency_ms = (time.time() - start) * 1000

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


print("\nEvaluating Q8 model...")
q8_res = evaluate_model(model_q8, Q8_PATH, "INT8")
print(f"Q8 → {q8_res['l2']:.3f} px | Size: {q8_res['size_str']} | Model Latency: {q8_res['model_latency_ms']:.2f} ms | Total Latency: {q8_res['total_latency_ms']:.2f} ms")







# ============================================================
# QAT
# ============================================================

print("\nStarting Quantization Aware Training (QAT)...")

# THIS LINES ARE FOR KERAS MODELS, DO NOT WORK FOR USING TORCH ONNX MODELS

# model_q8.compile(
#     loss=keras.losses.SparseCategoricalCrossentropy(from_logits=True),
#     optimizer=Adam(learning_rate=1e-4),
#     metrics=['accuracy']
# )

# model_q8.fit(x_train, y_train, epochs=5, validation_split=0.1)

print("QAT complete.")






# ============================================================
# SAVE QAT MODEL
# ============================================================

save_model(model_q8, QAT8_PATH)
print(f"\nQAT model saved:")
print(f"  QAT8 → {QAT8_PATH}")

print("\nVerifying saved QAT8 model structure...")
model_qat8_onnx = onnx.load(str(QAT8_PATH))
print("QAT8 ONNX model loaded")
print("Input shape:", [x.dim_value or x.dim_param for x in model_qat8_onnx.graph.input[0].type.tensor_type.shape.dim])
print("Output shape:", [x.dim_value or x.dim_param for x in model_qat8_onnx.graph.output[0].type.tensor_type.shape.dim])









# ============================================================
# EVALUATE QAT8 IN TEST DATASET (REAL PX L2 ERROR)
# ============================================================

print("\n" + "="*70)
print("EVALUATING QAT MODEL (L2 IN REAL PIXELS)")
print("="*70)

print("\nEvaluating QAT8 model...")
qat8_res = evaluate_model(model_q8, QAT8_PATH, "INT8")
print(f"QAT8 → {qat8_res['l2']:.3f} px | Size: {qat8_res['size_str']} | Model Latency: {qat8_res['model_latency_ms']:.2f} ms | Total Latency: {qat8_res['total_latency_ms']:.2f} ms")










# ============================================================
# Final results table (L2 error in pixels)
# ============================================================
print("\n" + "="*85)
print("FINAL QUANTIZATION AWARED TRAINING (QAT) RESULTS SUMMARY")
print("="*85)
print(f"{'Model':<12} {'Gaze L2 (px)':>18} {'Δ(px) vs Q8':>16} {'Size':>15} {'Reduction':>12} {'Model Lat(ms)':>15} {'Δ(%) vs Q8':>16} {'Total Lat(ms)':>15} {'Δ(%) vs Q8':>16}")
print("-"*85)

# Q8 baseline
print(f"{'Q8':<12} {q8_res['l2']:18.3f} {'N/A':>16} {q8_res['size_str']:>15} {'N/A':>12} {q8_res['model_latency_ms']:>15.2f} {'N/A':>16} {q8_res['total_latency_ms']:>15.2f} {'N/A':>16}")

# QAT8
delta_l2 = qat8_res['l2'] - q8_res['l2']
reduction_qat = "N/A" if q8_res['size_str'] == "N/A" or qat8_res['size_str'] == "N/A" else \
    f"{(1 - os.path.getsize(QAT8_PATH) / os.path.getsize(Q8_PATH))*100:5.1f}%"
delta_model_lat_8 = "N/A" if q8_res['model_latency_ms'] == 0 or qat8_res['model_latency_ms'] == 0 else \
    ( qat8_res['model_latency_ms'] - q8_res['model_latency_ms'] ) / q8_res['model_latency_ms']
delta_total_lat_8 = "N/A" if q8_res['total_latency_ms'] == 0 or qat8_res['total_latency_ms'] == 0 else \
    ( qat8_res['total_latency_ms'] - q8_res['total_latency_ms'] ) / q8_res['total_latency_ms']
print(f"{'QAT8':<12} {qat8_res['l2']:18.3f} {delta_l2:+16.3f} px {qat8_res['size_str']:>15} {reduction_qat:>12} {qat8_res['model_latency_ms']:>15.2f} {delta_model_lat_8:+15.2%} {qat8_res['total_latency_ms']:>15.2f} {delta_total_lat_8:+15.2%}")


print("-"*85)
print(f"Total test samples: {samples:,}")
print(f"Original Q8 model: {Q8_PATH}")
print(f"QAT model: {QAT8_PATH}")
print("="*85)







# ============================================================
# Save clean report (L2 only)
# ============================================================
summary_path = QUANTIZED_FOLDER_PATH / "qat_report.txt"
with open(summary_path, "w") as f:
    f.write("Quantization Aware TRaining (QAT) Report\n")
    f.write("="*65 + "\n")
    f.write(f"{'Model':<12} {'Gaze L2 (px)':>15} {'Δ(px) vs Q8':>15} {'Size':>15} {'Reduction':>12} {'Model Lat(ms)':>15} {'Δ(%) vs Q8':>16} {'Total Lat(ms)':>15} {'Δ(%) vs Q8':>16} \n")
    f.write("-"*65 + "\n")
    
    f.write(f"{'Tennst':<12} {q8_res['l2']:15.3f} {'N/A':>15} {q8_res['size_str']:>15} {'N/A':>12} {q8_res['model_latency_ms']:>15.2f} {'N/A':>16} {q8_res['total_latency_ms']:>15.2f} {'N/A':>16}\n")
    
    f.write(f"{'Q_INT8':<12} {qat8_res['l2']:15.3f} {delta_l2:+8.3f} px {qat8_res['size_str']:>15} {reduction_qat:>12} {qat8_res['model_latency_ms']:>15.2f} {delta_model_lat_8:+15.2%} {qat8_res['total_latency_ms']:>15.2f} {delta_total_lat_8:+15.2%}\n")
    
    f.write("-"*65 + "\n")
    f.write(f"Test samples: {samples:,}\n")
    f.write(f"Original Q8 model: {Q8_PATH}\n")
    f.write(f"QAT model: {QAT8_PATH}\n")
    f.write(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")

print(f"Clean L2 report saved → {summary_path}")

