# 4_test_tennst.py
# Evaluate trained EyeTennSt on the test set
# → Reports: Gaze MAE (px), State Accuracy (%), total samples
# → Works with preprocessed_fast/ [N,10,640,480] float16

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from pathlib import Path
import numpy as np
from tqdm import tqdm
from _2_model_f16 import EyeTennSt   # ← your model with .float() removed

# ============================================================
# CONFIG — UPDATE THESE PATHS
# ============================================================
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# CHANGE: Point to your new fast dataset
DATA_ROOT = Path("/home/dronelab-pc-1/Jon/IndustrialProject/akida_examples/1_ec_example/training/preprocessed_fast")

# Path to your trained model (best.pth or final.pth)
MODEL_PATH = Path("/home/dronelab-pc-1/Jon/IndustrialProject/akida_examples/1_ec_example/training/runs/tennst_3_f16_batch_24_epochs_130/best.pth")

BATCH_SIZE = 64   # large batch = fast inference
print(f"Testing on {DEVICE}")
print(f"Model: {MODEL_PATH.name}")
print(f"Dataset: {DATA_ROOT}")

# ============================================================
# SAME DATASET AS TRAINING — BUT TEST SPLIT ONLY
# ============================================================
class EyeTrackingDataset(Dataset):
    def __init__(self, split="test"):
        self.recordings = []
        self.samples = []
        split_path = DATA_ROOT / split

        for rec_dir in sorted(split_path.iterdir()):
            if not rec_dir.is_dir():
                continue
            voxels_path = rec_dir / "voxels.pt"      # [N, 10, 640, 480] float16
            labels_path = rec_dir / "labels.txt"
            if not voxels_path.exists() or not labels_path.exists():
                continue

            voxels = torch.load(voxels_path, map_location="cpu")  # [N, 10, 640, 480]
            labels = np.loadtxt(labels_path, dtype=np.int32)      # [N, 4]

            self.recordings.append({
                'voxels': voxels,
                'labels': labels,
                'num_frames': len(voxels)
            })

        # Build index
        for rec_idx, item in enumerate(self.recordings):
            for i in range(item['num_frames']):
                self.samples.append((rec_idx, i))

        print(f"TEST dataset: {len(self.samples)} samples from {len(self.recordings)} recordings")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        rec_idx, frame_idx = self.samples[idx]
        item = self.recordings[rec_idx]

        window = item['voxels'][frame_idx]                    # [10, 640, 480] float16
        x, y, state = item['labels'][frame_idx, 1:4]          # x, y, state

        return {
            'input': window,
            'gaze': torch.tensor([x, y], dtype=torch.float32),
            'state': torch.tensor(state, dtype=torch.float32)
        }

# ============================================================
# MAIN TEST LOOP
# ============================================================
@torch.no_grad()
def main():
    # Dataset + loader
    test_dataset = EyeTrackingDataset(split="test")
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False,
                             num_workers=8, pin_memory=True)

    # Load model
    torch.cuda.empty_cache()
    model = EyeTennSt(t_kernel_size=5, s_kernel_size=3, n_depthwise_layers=6).to(DEVICE)

    # DATA PARALLEL for multi-GPU setups
    if torch.cuda.device_count() > 1:
        print(f"Using {torch.cuda.device_count()} GPUs with DataParallel...")
        model = torch.nn.DataParallel(model)

    model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
    model.eval()
    print("Model loaded and set to eval mode")

    # Metrics
    total_gaze_error = 0.0
    total_state_correct = 0
    total_samples = 0

    print("Starting inference...")
    for batch in tqdm(test_loader, desc="Testing"):
        x = batch['input'].to(DEVICE, non_blocking=True)           # [B,10,640,480] float16
        gaze_target = batch['gaze'].to(DEVICE)
        state_target = batch['state'].to(DEVICE)

        # Forward pass (AMP for speed, optional but nice)
        with torch.cuda.amp.autocast():
            gaze_pred, state_logit = model(x)

        # Gaze: Mean Absolute Error in pixels
        gaze_error = torch.mean(torch.abs(gaze_pred - gaze_target), dim=1)  # per sample
        total_gaze_error += gaze_error.sum().item()

        # State: binary accuracy
        state_pred = (torch.sigmoid(state_logit) > 0.5).float()
        total_state_correct += (state_pred == state_target).sum().item()

        total_samples += x.size(0)

    # Final results
    avg_gaze_mae = total_gaze_error / total_samples
    state_accuracy = total_state_correct / total_samples * 100

    print("\n" + "="*50)
    print("FINAL TEST RESULTS")
    print("="*50)
    print(f"Total test samples: {total_samples}")
    print(f"Gaze MAE:           {avg_gaze_mae:.3f} pixels")
    print(f"Eye State Accuracy: {state_accuracy:.2f}%")
    print("="*50)

    # Save results to txt
    result_file = MODEL_PATH.parent / "test_results.txt"
    with open(result_file, 'w') as f:
        f.write(f"Gaze MAE: {avg_gaze_mae:.3f} px\n")
        f.write(f"State Accuracy: {state_accuracy:.2f}%\n")
        f.write(f"Total samples: {total_samples}\n")
    print(f"Results saved to: {result_file}")

if __name__ == "__main__":
    main()