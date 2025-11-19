# 5_quantize.py
# Quantize trained EyeTennSt (PyTorch) → int8 / int4 for Akida SNN conversion
# Uses quantizeml (official BrainChip tool) — the same as in Akida examples

import torch
from torch.utils.data import DataLoader
from pathlib import Path
import numpy as np
from tqdm import tqdm

# Akida quantization tools
from quantizeml.models import quantize
from quantizeml.layers import QuantizationParams

# Your model and dataset
from training._2_model_f16 import EyeTennSt
from training._3_train_even_faster import EyeTrackingDataset  # ← use the fast one we made

# ============================================================
# CONFIG — UPDATE THESE
# ============================================================
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Paths
MODEL_PATH = Path("training/runs/tennst_3_f16_batch_64_epochs_150/best.pth")           # your trained float16 model
DATA_ROOT = Path("/home/dronelab-pc-1/Jon/IndustrialProject/akida_examples/1_ec_example/training/preprocessed_fast")

# Output
QUANTIZED_FOLDER_PATH = Path("quantized_models")
QUANTIZED_FOLDER_PATH.mkdir(exist_ok=True)

print(f"Loading model from: {MODEL_PATH}")
print(f"Using calibration data from: {DATA_ROOT/'train'}")

# ============================================================
# 1. Load trained PyTorch model and convert to Keras
# ============================================================
# Load PyTorch weights
state_dict = torch.load(MODEL_PATH, map_location="cpu")
pytorch_model = EyeTennSt(t_kernel_size=5, s_kernel_size=3, n_depthwise_layers=6)
pytorch_model.load_state_dict(state_dict)
pytorch_model.eval()

# Export to Keras via TorchScript → ONNX → Keras (quantizeml handles this)
# quantizeml accepts PyTorch models directly via torch.export or ONNX
# We use the recommended way: export to ONNX first
print("Exporting PyTorch model to ONNX...")
dummy_input = torch.randn(1, 10, 640, 480, device="cpu", dtype=torch.float16)  # matches training
TMP_QUANTIZED_MODEL_PATH = f"{QUANTIZED_FOLDER_PATH}/tmp_eyetennst.onnx"
torch.onnx.export(
    pytorch_model,
    dummy_input,
    TMP_QUANTIZED_MODEL_PATH,
    export_params=True,
    opset_version=17,
    do_constant_folding=True,
    input_names=['input'],
    output_names=['gaze', 'state'],
    dynamic_axes={'input': {0: 'batch'}}
)
print("ONNX export done → tmp_eyetennst.onnx")

# Load as Keras model (quantizeml can read ONNX directly)
from quantizeml.models import load_model
model_keras = load_model(TMP_QUANTIZED_MODEL_PATH)
print("Keras model loaded from ONNX")

# ============================================================
# 2. Prepare calibration dataset (1000–2000 samples recommended)
# ============================================================
calib_dataset = EyeTrackingDataset(split="train")
calib_loader = DataLoader(calib_dataset, batch_size=64, shuffle=True, num_workers=8)

calibration_samples = []
print("Collecting calibration samples...")
for i, batch in enumerate(tqdm(calib_loader)):
    if i >= 20:  # ~1280 samples
        break
    x = batch['input']  # [B, 10, 640, 480] float16
    calibration_samples.append(x.numpy())  # quantizeml wants numpy

calibration_samples = np.concatenate(calibration_samples, axis=0)
print(f"Collected {len(calibration_samples)} calibration samples")

# ============================================================
# 3. QUANTIZATION OPTIONS — Choose one
# ============================================================

# OPTION A: 8-bit quantization (best accuracy, good for Akida)
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

# OPTION B: 4-bit (smaller, for maximum efficiency)
print("\nQuantizing to 4-bit...")
qparams_4bit = QuantizationParams(
   input_weight_bits=8,
   weight_bits=4,
   activation_bits=4,
   per_tensor_activations=True
)
model_q4 = quantize(model_keras, qparams=qparams_4bit,
                     samples=calibration_samples, num_samples=1024)

# ============================================================
# 4. Save quantized model
# ============================================================

INT4_QUANTIZED_MODEL_PATH = f"{QUANTIZED_FOLDER_PATH}/q_tennst_int4.keras"
INT8_QUANTIZED_MODEL_PATH = f"{QUANTIZED_FOLDER_PATH}/q_tennst_int8.keras"

model_q4.save(INT4_QUANTIZED_MODEL_PATH)
print(f"\nQuantized model saved to: {INT4_QUANTIZED_MODEL_PATH}")

model_q8.save(INT8_QUANTIZED_MODEL_PATH)
print(f"\nQuantized model saved to: {INT8_QUANTIZED_MODEL_PATH}")


# ============================================================
# 5. Test accuracy drop — compare float vs int8 vs int4 on TEST set
# ============================================================
print("\n" + "="*60)
print("EVALUATING ACCURACY DROP ON TEST SET")
print("="*60)

import tensorflow as tf
from tensorflow.keras.metrics import MeanAbsoluteError, BinaryAccuracy

# Re-use the same fast test dataset
test_dataset = EyeTrackingDataset(split="test")
test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False, num_workers=8, pin_memory=True)

# ----------------------------------
# Helper: evaluate any quantizeml/Keras model and return metrics
# ----------------------------------
def evaluate_quantized_model(model_keras, model_name: str):
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

    return {
        "name": model_name,
        "mae": gaze_mae,
        "acc": blink_acc,
        "samples": total_samples
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

        with torch.cuda.amp.autocast():
            gaze_pred, state_logit = pytorch_model(x)

        # Gaze MAE
        mae_metric.update_state(gaze_true.cpu().numpy(), gaze_pred.cpu().numpy())
        # Blink accuracy
        state_pred = (torch.sigmoid(state_logit) > 0.5).float()
        bin_acc.update_state(state_true.cpu().numpy(), state_pred.cpu().numpy())

        float_results["samples"] += x.size(0)

float_results["mae"] = mae_metric.result().numpy()
float_results["acc"] = bin_acc.result().numpy() * 100.0

# ----------------------------------
# 2. Evaluate quantized models
# ----------------------------------
int8_results = evaluate_quantized_model(model_q8, "INT8")
int4_results = evaluate_quantized_model(model_q4, "INT4")

# ----------------------------------
# 3. Print beautiful real table
# ----------------------------------
results = [float_results, int8_results, int4_results]

print("\n" + "="*80)
print("FINAL QUANTIZATION ACCURACY COMPARISON")
print("="*80)
print(f"{'Model':<18} {'Gaze MAE (px)':>14} {'Δ MAE':>10} {'Blink Acc (%)':>16} {'Δ Acc':>10}")
print("-"*80)

for i, r in enumerate(results):
    if i == 0:
        delta_mae = 0.0
        delta_acc = 0.0
    else:
        delta_mae = r["mae"] - results[0]["mae"]
        delta_acc = r["acc"] - results[0]["acc"]

    print(f"{r['name']:<18} "
          f"{r['mae']:14.3f} "
          f"{delta_mae:+8.3f} "
          f"{r['acc']:14.2f}% "
          f"{delta_acc:+8.2f} pp")

print("-"*80)
print(f"Total test samples: {results[0]['samples']:,}")
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
            f.write(f"{r['name']}:  Gaze MAE = {r['mae']:.3f} px | Blink Acc = {r['acc']:.2f}%\n")
        else:
            delta_mae = r["mae"] - results[0]["mae"]
            delta_acc = r["acc"] - results[0]["acc"]
            f.write(f"{r['name']}:  Gaze MAE = {r['mae']:.3f} px ({delta_mae:+.3f}) | "
                    f"Blink Acc = {r['acc']:.2f}% ({delta_acc:+.2f} pp)\n")
print(f"Report saved to: {summary_path}")
