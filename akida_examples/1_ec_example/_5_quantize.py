# _5_quantize_akida.py
# Quantize trained EyeTennSt (Akida-style) → int8 / int4 for Akida SNN conversion
# Now works with: uint8 [2,50,96,128] input + heatmap [3,50,3,4] output
# Uses quantizeml (official BrainChip tool)

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from pathlib import Path
import numpy as np
from tqdm import tqdm

# Akida quantization tools
from quantizeml.models import quantize, load_model
from quantizeml.layers import QuantizationParams

# Model and dataset
from training._2_5_model_uint8_akida import EyeTennSt
from training._3_4_train_fast_akida import AkidaGazeDataset, get_out_gaze_point 

# ============================================================
# CONFIG 
# ============================================================
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Paths
MODEL_PATH = Path("training/runs/tennst_11_akida_b128_epochs_100/best.pth") 
DATA_ROOT = Path("/home/dronelab-pc-1/Jon/IndustrialProject/akida_examples/1_ec_example/training/preprocessed_fast")

# Output
QUANTIZED_FOLDER_PATH = Path("quantized_models")
QUANTIZED_FOLDER_PATH.mkdir(exist_ok=True)

print(f"Loading model from: {MODEL_PATH}")
print(f"Using calibration data from: {DATA_ROOT/'train'}")

# ============================================================
# 1. Load trained PyTorch model
# ============================================================

state_dict = torch.load(MODEL_PATH, map_location="cpu")
model = EyeTennSt(t_kernel_size=5, s_kernel_size=3, n_depthwise_layers=4).eval()
model.load_state_dict(state_dict if "model_state_dict" not in state_dict else state_dict["model_state_dict"])
print("PyTorch model loaded")

# ============================================================
# 2. Export to ONNX (quantizeml accepts ONNX directly)
# ============================================================

print("Exporting to ONNX...")
dummy_input = torch.randint(0, 256, (1, 2, 50, 96, 128), dtype=torch.uint8, device="cpu")  # ← uint8 input!

onnx_path = QUANTIZED_FOLDER_PATH / "eyetennst_akida.onnx"
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

# Load as Keras model
model_keras = load_model(str(onnx_path))
print("Keras model loaded from ONNX")

# ============================================================
# 3. Calibration dataset (from preprocessed_akida)
# ============================================================

calib_dataset = AkidaGazeDataset(split="train")
calib_loader = DataLoader(calib_dataset, batch_size=64, shuffle=True, num_workers=8, pin_memory=True)

calibration_samples = []
print("Collecting calibration samples (uint8 voxels)...")
for i, batch in enumerate(tqdm(calib_loader, total=25)):
    if i >= 25:  # ~1600 samples
        break
    x = batch['input'].numpy()    # ← [B,2,50,96,128] uint8 → numpy
    calibration_samples.append(x)
calibration_samples = np.concatenate(calibration_samples, axis=0)
print(f"Collected {len(calibration_samples)} calibration samples")

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
print("EVALUATING QUANTIZED MODELS (Gaze MAE in pixels)")
print("="*70)

test_dataset = AkidaGazeDataset(split="test")
test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False, num_workers=8, pin_memory=True)

def evaluate_model(model_keras, name, path=None):
    total_error_px = 0.0
    total_samples = 0
    pbar = tqdm(test_loader, desc=f"Eval {name}", leave=False)
    for batch in pbar:
        x_np = batch['input'].numpy()           # [B,2,50,96,128] uint8
        target = batch['target'].numpy()        # [B,3,50,3,4] float32

        # Keras inference
        pred_heatmap = model_keras.predict(x_np, verbose=0)  # [B,3,50,3,4]

        # Convert to torch for get_out_gaze_point
        pred_t = torch.from_numpy(pred_heatmap)
        targ_t = torch.from_numpy(target)

        pred_x, pred_y, _, _ = get_out_gaze_point(pred_t)
        gt_x,   gt_y,   _, _ = get_out_gaze_point(targ_t)

        # Convert to real pixels
        pred_px = pred_x * 160.0
        pred_py = pred_y * 160.0
        gt_px   = gt_x   * 160.0
        gt_py   = gt_y   * 160.0

        error_px = torch.sqrt((pred_px - gt_px)**2 + (pred_py - gt_py)**2)
        total_error_px += error_px.sum().item()
        total_samples += x_np.shape[0]

    mae_px = total_error_px / total_samples
    size_str = format_size(os.path.getsize(path)) if path and path.exists() else "N/A"
    return {"name": name, "mae": mae_px, "size_str": size_str, "samples": total_samples}

def format_size(b):
    for unit in ['B', 'KB', 'MB', 'GB']:
        if b < 1024: return f"{b:.1f}{unit}"
        b /= 1024
    return f"{b:.1f}GB"













# ============================================================
# 5. Test accuracy drop — compare float vs int8 vs int4 on TEST set
# ============================================================
print("\n" + "="*60)
print("EVALUATING ACCURACY DROP ON TEST SET")
print("="*60)

import tensorflow as tf
from tensorflow.keras.metrics import MeanAbsoluteError, BinaryAccuracy
import os

# Re-use the same fast test dataset
test_dataset = EyeTrackingDataset(split="test")
test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False, num_workers=8, pin_memory=True)

# ----------------------------------
# Helper: get file size in human-readable format
# ----------------------------------
def format_size(bytes_size):
    for unit in ['B', 'KB', 'MB', 'GB']:
        if bytes_size < 1024:
            return f"{bytes_size:.2f} {unit}"
        bytes_size /= 1024
    return f"{bytes_size:.2f} GB"

# ----------------------------------
# Helper: evaluate any quantizeml/Keras model and return metrics
# ----------------------------------
def evaluate_quantized_model(model_keras, model_name: str, model_path: str = None):
    mae_metric = MeanAbsoluteError()
    bin_acc    = BinaryAccuracy(threshold=0.5)

    total_samples = 0
    pbar = tqdm(test_loader, desc=f"Evaluating {model_name}", leave=False)

    for batch in pbar:
        x_np          = batch['input'].numpy()                 # [B,10,640,480] float16 → numpy
        gaze_true     = batch['gaze'].numpy()                  # [B,2]
        state_true    = batch['state'].numpy().astype(np.float32)  # [B]

        # Keras inference (returns two outputs: gaze, state_logit)
        gaze_pred, state_logit = model_keras.predict(x_np, verbose=0)
        state_pred = (state_logit.squeeze() > 0).astype(np.float32)

        mae_metric.update_state(gaze_true, gaze_pred)
        bin_acc.update_state(state_true, state_pred)

        total_samples += x_np.shape[0]

    gaze_mae   = mae_metric.result().numpy()
    blink_acc  = bin_acc.result().numpy() * 100.0

    size_bytes = os.path.getsize(model_path) if model_path and os.path.exists(model_path) else 0

    return {
        "name": model_name,
        "mae": gaze_mae,
        "acc": blink_acc,
        "samples": total_samples,
        "size_bytes": size_bytes,
        "size_str": format_size(size_bytes) if size_bytes > 0 else "N/A"
    }

# ----------------------------------
# 1. Evaluate Float16 baseline (using the original PyTorch model)
# ----------------------------------
print("\nEvaluating original Float16 model (PyTorch)...")
pytorch_model.to(DEVICE)
pytorch_model.eval()

float_results = {"name": "Float16 (orig)", "mae": 0.0, "acc": 0.0, "samples": 0}
mae_metric = tf.keras.metrics.MeanAbsoluteError()
bin_acc    = tf.keras.metrics.BinaryAccuracy(threshold=0.5)

with torch.no_grad():
    for batch in tqdm(test_loader, desc="Float16 baseline", leave=False):
        x = batch['input'].to(DEVICE, non_blocking=True)
        gaze_true = batch['gaze'].to(DEVICE)
        state_true = batch['state'].to(DEVICE)

        # with torch.cuda.amp.autocast():
        with torch.amp.autocast('cuda'):
            gaze_pred, state_logit = pytorch_model(x)

        # Gaze MAE
        mae_metric.update_state(gaze_true.cpu().numpy(), gaze_pred.cpu().numpy())
        # Blink accuracy
        state_pred = (torch.sigmoid(state_logit) > 0.5).float()
        bin_acc.update_state(state_true.cpu().numpy(), state_pred.cpu().numpy())

        float_results["samples"] += x.size(0)

float_results["mae"] = mae_metric.result().numpy()
float_results["acc"] = bin_acc.result().numpy() * 100.0
float_results["size_str"] = format_size(float_results["size_bytes"])

# ----------------------------------
# 2. Evaluate quantized models

# ----------------------------------
int8_results = evaluate_quantized_model(model_q8, "INT8", INT8_QUANTIZED_MODEL_PATH)
int4_results = evaluate_quantized_model(model_q4, "INT4", INT4_QUANTIZED_MODEL_PATH)

# ----------------------------------
# 3. Print beautiful real table
# ----------------------------------
results = [float_results, int8_results, int4_results]

print("\n" + "="*80)
print("FINAL QUANTIZATION RESULTS (Accuracy + Model Size)")
print("="*80)
print(f"{'Model':<14} {'Gaze MAE (px)':>14} {'Δ MAE':>10} {'Blink Acc':>14} {'Δ Acc':>10} {'Size':>12} {'Reduction':>12}")
print("-"*80)

for i, r in enumerate(results):
    if i == 0:
        delta_mae = 0.0
        delta_acc = 0.0
        reduction = "100.0%"
    else:
        delta_mae = r["mae"] - results[0]["mae"]
        delta_acc = r["acc"] - results[0]["acc"]
        reduction = f"{(1 - r['size_bytes']/results[0]['size_bytes'])*100:5.1f}%"

    print(f"{r['name']:<14} "
          f"{r['mae']:14.3f} "
          f"{delta_mae:+8.3f} "
          f"{r['acc']:14.2f}% "
          f"{delta_acc:+8.2f} pp "
          f"{r['size_str']:>12} "
          f"{reduction:>10}")

print("-"*80)
print(f"Total test samples: {results[0]['samples']:,}")
print(f"Reference (Float16): {float_results['size_str']}")
print("="*80)

# ----------------------------------
# 4. Save results to file (optional)
# ----------------------------------
summary_path = QUANTIZED_FOLDER_PATH / "quantization_report.txt"
with open(summary_path, "w") as f:
    f.write("Quantization Accuracy Report\n")
    f.write("="*50 + "\n")
    for r in results:
        if r is float_results:
            f.write(f"{r['name']:12} | MAE: {r['mae']:.3f} px | Acc: {r['acc']:.2f}% | Size: {r['size_str']}\n")
        else:
            delta_mae = r["mae"] - results[0]["mae"]
            delta_acc = r["acc"] - results[0]["acc"]
            reduction = (1 - r['size_bytes']/results[0]['size_bytes'])*100
            f.write(f"{r['name']:12} | MAE: {r['mae']:.3f} px (+{delta_mae:.3f}) | "
                    f"Acc: {r['acc']:.2f}% ({delta_acc:+.2f} pp) | "
                    f"Size: {r['size_str']} (-{reduction:.1f}%)\n")
print(f"Report saved to: {summary_path}")
