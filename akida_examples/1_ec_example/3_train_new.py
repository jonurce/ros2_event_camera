# 3_train.py
import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np
from pathlib import Path

class EventEyeDataset10Bins(Dataset):
    def __init__(self, root_dir="preprocessed", split="train"):
        self.root = Path(root_dir) / split
        self.samples = []
        
        for rec_dir in sorted(self.root.iterdir()):
            if not rec_dir.is_dir():
                continue
            pt_path = rec_dir / "voxels.pt"
            label_path = rec_dir / "labels.txt"
            if not pt_path.exists() or not label_path.exists():
                continue
                
            voxels = torch.load(pt_path)  # [N, 640, 480] int8
            labels = np.loadtxt(label_path, dtype=np.int32)  # [N, 4]: t, x, y, state
            
            for i in range(len(voxels)):
                self.samples.append({
                    'voxels': voxels,
                    'labels': labels,
                    'idx': i,
                    'rec_id': rec_dir.name
                })

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        item = self.samples[idx]
        i = item['idx']
        voxels = item['voxels']  # [N, 640, 480]
        labels = item['labels']

        # === Build 10-bin window: from i-9 to i ===
        start_idx = max(0, i - 9)
        end_idx = i + 1
        window = voxels[start_idx:end_idx]  # [up to 10, 640, 480]

        # Pad with zeros if at beginning
        if window.shape[0] < 10:
            pad = 10 - window.shape[0]
            padding = torch.zeros(pad, 640, 480, dtype=window.dtype)
            window = torch.cat([padding, window], dim=0)

        # Final input: [10, 640, 480]
        x = window  # int8 → will convert to float later if needed

        # Targets
        y_reg = torch.tensor([labels[i, 1], labels[i, 2]], dtype=torch.float32)  # x, y
        y_state = torch.tensor(labels[i, 3], dtype=torch.long)                   # 0 or 1

        return {
            'input': x,           # [10, 640, 480] int8
            'gaze': y_reg,        # [2]
            'state': y_state      # scalar
        }
    

dataset = EventEyeDataset10Bins(split="train")
loader = DataLoader(dataset, batch_size=16, shuffle=True, num_workers=4, pin_memory=True)

for batch in loader:
    x = batch['input'].float().to(device)    # [B, 10, 640, 480] → float32 on GPU
    gaze = batch['gaze'].to(device)          # [B, 2]
    state = batch['state'].to(device)        # [B]

    # Add channel dim if model expects [B, C, T, H, W]
    x = x.unsqueeze(2)  # → [B, 10, 1, 640, 480] or later permute



criterion_gaze  = nn.MSELoss()
criterion_state = nn.BCEWithLogitsLoss()   # ← important: WithLogits!

gaze_pred, state_logit = model(x)
loss_gaze  = criterion_gaze(gaze_pred, gaze_target)                    # [B,2]
loss_state = criterion_state(state_logit, state_target.float())        # state_target: 0 or 1
loss = loss_gaze + 5.0 * loss_state   # you can weight them