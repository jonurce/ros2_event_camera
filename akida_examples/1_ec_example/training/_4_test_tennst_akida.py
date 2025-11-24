# _4_test_tennst_akida.py
# Evaluate the trained Akida-exact EyeTennSt model on the test set
# → Reports: Gaze MAE in pixels (real px), total samples
# → Uses the new preprocessed_akida dataset: uint8 voxels + float32 heatmaps [3,50,3,4]
# → Uses soft-argmax + correct pixel scaling (160 px per cell)

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from pathlib import Path
from tqdm import tqdm

# ←←← IMPORT YOUR AKIDA MODEL AND DATASET ←←←
from _2_5_model_uint8_akida import EyeTennSt
from _3_4_train_fast_akida import AkidaGazeDataset, extract_gaze

# Performance settings
torch.backends.cuda.matmul.allow_tf32 = False
torch.backends.cudnn.benchmark = False
import os
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

# ============================================================
# CONFIG
# ============================================================
DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# ←←← CHANGE THIS: your new preprocessed_akida folder
DATA_ROOT = Path("/home/dronelab-pc-1/Jon/IndustrialProject/akida_examples/1_ec_example/training/preprocessed_akida")

# ←←← CHANGE THIS: your best.pth from the new training run
MODEL_PATH = Path("/home/dronelab-pc-1/Jon/IndustrialProject/akida_examples/1_ec_example/training/runs/tennst_8_akida_b64_e100/best.pth")

BATCH_SIZE = 128        # even larger = faster inference (uint8 input = tiny memory)
W, H = 640, 480         # original image size
W_IN, H_IN = 128, 96    # Akida input size
W_OUT, H_OUT = 4, 3     # final feature map size
PIXEL_SCALE = (W_IN + H_IN) / (W_OUT + H_OUT)

print(f"Testing Akida-exact model on {DEVICE}")
print(f"Model: {MODEL_PATH.name}")
print(f"Dataset: {DATA_ROOT}")



# ============================================================
# MAIN TEST LOOP
# ============================================================
@torch.no_grad()
def main():
    # Dataset + loader
    test_dataset = AkidaGazeDataset(split="test")    
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE,
                             shuffle=False, num_workers=12,
                             pin_memory=True, persistent_workers=True)

    # Model
    torch.cuda.empty_cache()
    model = EyeTennSt(t_kernel_size=5, s_kernel_size=3, n_depthwise_layers=4).to(DEVICE)

    # Multi-GPU support
    if torch.cuda.device_count() > 1:
        print(f"→ Using {torch.cuda.device_count()} GPUs with DataParallel")
        model = torch.nn.DataParallel(model)

    # Compile for speed
    print("Compiling model with torch.compile()...")
    model = torch.compile(model, mode="reduce-overhead")

    # Load weights
    checkpoint = torch.load(MODEL_PATH, map_location=DEVICE)

    if 'model_state_dict' in checkpoint:
        state_dict = checkpoint['model_state_dict']
        print(f"Loaded checkpoint from epoch {checkpoint.get('epoch', '?')}")
        print(f"   → val_loss: {checkpoint.get('val_loss', 'N/A'):.4f} | val_mae_px: {checkpoint.get('val_mae_px', 'N/A'):.3f}")
    else:
        state_dict = checkpoint  # old format: direct state_dict
    
    # Load into model (handles DataParallel correctly)
    if isinstance(model, torch.nn.DataParallel):
        model.module.load_state_dict(state_dict)
    else:
        model.load_state_dict(state_dict)
    
    model.eval()
    print("Model loaded and ready")

    # Metrics
    total_gaze_error_px = 0.0
    total_samples = 0

    print("Starting inference on test set...")
    for batch in tqdm(test_loader, desc="Testing"):
        x = batch['input'].to(DEVICE, non_blocking=True)      # [B,2,50,96,128] uint8
        heatmap_target = batch['target'].to(DEVICE, non_blocking=True)  # [B,3,50,3,4]

        with torch.amp.autocast('cuda'):
            pred = model(x)                                   # [B,3,50,3,4]

        # Extract predicted gaze in normalized coordinates
        gaze_pred_norm = extract_gaze(pred)                   # [B,2] in [0,1]

        # Extract GT gaze using SAME soft-argmax (for fair comparison)
        gaze_gt_norm = extract_gaze(heatmap_target)

        # Convert to real pixels
        gaze_pred_px = gaze_pred_norm * torch.tensor([W_IN, H_IN], device=DEVICE)
        gaze_gt_px   = gaze_gt_norm   * torch.tensor([W_IN, H_IN], device=DEVICE)

        # MAE in pixels
        error_px = torch.abs(gaze_pred_px - gaze_gt_px).sum(dim=1)   # L1 per sample
        total_gaze_error_px += error_px.sum().item()
        total_samples += x.size(0)

    # Final results
    avg_mae_px = total_gaze_error_px / total_samples

    print("\n" + "="*60)
    print("FINAL AKIDA-EXACT TEST RESULTS")
    print("="*60)
    print(f"Total test samples : {total_samples:,}")
    print(f"Gaze MAE           : {avg_mae_px:.3f} pixels")
    print(f"Pixel scale factor : {PIXEL_SCALE:.1f} px per cell")
    print("="*60)

    # Save results
    result_file = MODEL_PATH.parent / "test_results.txt"
    with open(result_file, 'w') as f:
        f.write(f"Gaze MAE: {avg_mae_px:.3f} px\n")
        f.write(f"Total samples: {total_samples}\n")
        f.write(f"Dataset: {DATA_ROOT}\n")
        f.write(f"Model: {MODEL_PATH.name}\n")
    print(f"Results saved to: {result_file}")

if __name__ == "__main__":
    main()