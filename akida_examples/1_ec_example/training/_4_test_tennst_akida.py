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
from _3_4_train_fast_akida import AkidaGazeDataset, get_out_gaze_point

# Performance settings
torch.backends.cuda.matmul.allow_tf32 = False
torch.backends.cudnn.benchmark = False
import os
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

# ============================================================
# CONFIG
# ============================================================
DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# ←←← CHANGE THIS: your best.pth from the new training run
MODEL_PATH = Path("/home/dronelab-pc-1/Jon/IndustrialProject/akida_examples/1_ec_example/training/runs/tennst_16_akida_b128_e150_ce_12mse_origpx/best.pth") 

BATCH_SIZE = 128        # even larger = faster inference (uint8 input = tiny memory)
W, H = 640, 480         # original image size
W_IN, H_IN = 128, 96    # Akida input size
W_OUT, H_OUT = 4, 3     # final feature map size

print(f"Testing Akida-exact model on {DEVICE}")
print(f"Model: {MODEL_PATH.name}")


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
        print(f"   →  val_loss_l2: {checkpoint.get('val_loss_l2', 'N/A'):.3f}")
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
    n_test   = len(test_loader)

    print("Starting inference on test set...")
    for batch in tqdm(test_loader, desc="Testing"):
        x = batch['input'].to(DEVICE, non_blocking=True)      # [B,2,50,96,128] uint8
        target = batch['target'].to(DEVICE, non_blocking=True)  # [B,3,50,3,4]

        with torch.amp.autocast('cuda'):
            pred = model(x)                                   # [B,3,50,3,4]

        # Extract predicted gaze in out coordinates
        pred_x, pred_y, pred_cell, pred_conf = get_out_gaze_point(pred)

        # Extract target gaze in out coordinates
        gt_x, gt_y, gt_cell, gt_conf = get_out_gaze_point(target)

        # pred_x, pred_y ∈ [0,4) and [0,3) → grid coordinates
        # Convert to model input pixels (128x96)
        pred_px = pred_x * (W / W_OUT)  
        pred_py = pred_y * (H / H_OUT)  
        gt_px   = gt_x   * (W / W_OUT)
        gt_py   = gt_y   * (H / H_OUT)

        # Now L2 in real pixels
        loss_l2_px = torch.sqrt((pred_px - gt_px)**2 + (pred_py - gt_py)**2).mean()

        # Accumulate real pixel error
        total_gaze_error_px += loss_l2_px.item()
        total_samples += x.size(0)

    # Final results
    # avg_l2_px = total_gaze_error_px / total_samples
    avg_l2_px = total_gaze_error_px / n_test

    print("\n" + "="*60)
    print("FINAL AKIDA-EXACT TEST RESULTS")
    print("="*60)
    print(f"Total test samples          : {total_samples:,}")
    print(f"Total test batches         : {n_test:,}")
    print(f"Gaze L2 (in original px) : {avg_l2_px:.3f} pixels")
    print("="*60)

    # Save results
    result_file = MODEL_PATH.parent / "test_results.txt"
    with open(result_file, 'w') as f:
        f.write(f"Gaze L2 (in original px): {avg_l2_px:.3f} px\n")
        f.write(f"Total samples: {total_samples}\n")
        f.write(f"Model: {MODEL_PATH.name}\n")
    print(f"Results saved to: {result_file}")

if __name__ == "__main__":
    main()