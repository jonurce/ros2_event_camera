
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent / "training"))
from training._3_4_train_fast_akida import get_out_gaze_point, AkidaGazeDataset
import numpy as np
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





# ============================================================
# LOAD SAVED SNN INT8 MODEL
# ============================================================

AKIDA_FOLDER_PATH = Path("akida_examples/1_ec_example/akida_models")
akida_path = AKIDA_FOLDER_PATH / "akida2_int8.fbz"
akida_model = akida.Model(str(akida_path))
# akida_model = cnn2snn.load_akida_model(str(akida_path))
print("\nLoaded Akida SNN model:")
akida_model.summary()






# ============================================================
# INFERENCE WITH SNN MODEL
# ============================================================


# To check performance of akida_model, take note of the following:
#   - Input: must be numpy-raw, with an 8-bit unsigned integer data type (uint8)
#   - Input shape: the channel dimension must be in the last dimension [96, 128, 2]
#   - Output shape:  [3, 4, 3] 


# Example: one frame [96, 128, 2] uint8
B = 1
frame = np.random.randint(0, 1, (B, 96, 128, 2), dtype=np.int8)

# Run on Akida SNN
prediction = akida_model.predict(frame)  # [B, H=3, W=4, C=3]
print("Raw Akida output:", prediction.shape)

# Convert to gaze point (same as before)
pred_t = torch.from_numpy(prediction) 
pred_t = pred_t.permute(0, 3, 1, 2)    # [B, C=3, H=3, W=4]
pred_x, pred_y, _, conf = get_out_gaze_point(pred_t, one_frame_only=True)
gaze_x = pred_x * W / W_OUT
gaze_y = pred_y * H / H_OUT
print(f"Gaze: ({gaze_x[-1]:.1f}, {gaze_y[-1]:.1f}) px | Confidence: {conf[-1]:.3f}")









# ============================================================
# EVALUATE AKIDA SNN MODEL
# ============================================================

# akida_accuracy = akida_model.evaluate(x_test_raw, labels_test)
# print(f'Akida model accuracy: {100 * akida_accuracy:.2f} %')

test_dataset = AkidaGazeDataset(split="test")
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False, num_workers=4)

total_error_px = 0.0
total_samples = 0

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

    # Akida inference
    pred = akida_model.predict(x)          # [B, H=3, W=4, C=3]

    # Convert to torch for your gaze function
    pred_t = torch.from_numpy(pred)
    pred_t = pred_t.permute(0, 3, 1, 2)    # [B, C=3, H=3, W=4]
    targ_t = torch.from_numpy(target)

    pred_x, pred_y, _, _ = get_out_gaze_point(pred_t, one_frame_only=True)
    gt_x,   gt_y,   _, _ = get_out_gaze_point(targ_t, one_frame_only=True)

    # Convert to pixels
    pred_px = pred_x * W / W_OUT
    pred_py = pred_y * H / H_OUT
    gt_px   = gt_x   * W / W_OUT
    gt_py   = gt_y   * H / H_OUT

    # L2 error
    error_l2 = torch.sqrt((pred_px - gt_px)**2 + (pred_py - gt_py)**2)
    total_error_px += error_l2.sum().item()
    total_samples += error_l2.shape[0]

    pbar.set_postfix({"L2": total_error_px / total_samples})

# Final result
akida_l2 = total_error_px / total_samples
print(f"\nAKIDA SNN FINAL RESULT → Gaze L2 Error: {akida_l2:.3f} px on {total_samples} samples")