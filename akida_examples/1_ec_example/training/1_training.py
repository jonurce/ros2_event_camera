#%% ##################### 2 - Network architecture ######################
# Instantiating the spatiotemporal blocks]

# Show how to load and create the model
import torch
import torch.nn as nn

# Limit CPU threads
import os
os.environ['OMP_NUM_THREADS'] = '1'  
os.environ['MKL_NUM_THREADS'] = '1'
torch.set_num_threads(1)
torch.set_num_interop_threads(1)

from tenns_modules import SpatioTemporalBlock
from torchinfo import summary

n_depthwise_layers = 4
channels = [2, 8, 16, 32, 48, 64, 80, 96, 112, 128, 256]
t_kernel_size = 5  # can vary from 1 to 10
s_kernel_size = 3  # can vary in [1, 3, 5, 7] (1 only when depthwise is False)


class TennSt(nn.Module):
    def __init__(self, channels, t_kernel_size, s_kernel_size, n_depthwise_layers):
        super().__init__()

        depthwises = [False] * (10 - n_depthwise_layers) + [True] * n_depthwise_layers
        self.backbone = nn.Sequential()
        for i in range(0, len(depthwises), 2):
            in_channels, med_channels, out_channels = channels[i], channels[i + 1], channels[i + 2]
            t_depthwise, s_depthwise = depthwises[i], depthwises[i]

            self.backbone.append(
                SpatioTemporalBlock(in_channels=in_channels, med_channels=med_channels,
                                    out_channels=out_channels, t_kernel_size=t_kernel_size,
                                    s_kernel_size=s_kernel_size, s_stride=2, bias=False,
                                    t_depthwise=t_depthwise, s_depthwise=s_depthwise))

        self.head = nn.Sequential(
            SpatioTemporalBlock(channels[-1], channels[-1], channels[-1],
                                t_kernel_size=t_kernel_size, s_kernel_size=s_kernel_size,
                                t_depthwise=False, s_depthwise=False),
            nn.Conv3d(channels[-1], 3, 1)
        )

    def forward(self, input):
        return self.head((self.backbone(input)))


model = TennSt(channels, t_kernel_size, s_kernel_size, n_depthwise_layers)
summary(model, input_size=(1, 2, 50, 96, 128), depth=4, verbose=0)



#%% ##################### 3 - Dataset ######################

import numpy as np
from torch.utils.data import Dataset, DataLoader
from akida_models import fetch_file

# Fetch dataset
data_file = fetch_file("https://data.brainchip.com/dataset-mirror/eye_tracking_ais2024_cvpr/eye_tracking_preprocessed_400frames_test.npz", fname="eye_tracking.npz")
data = np.load(data_file)
events = data['events']  # (N, C=2, T=50, H=60, W=80)
centers = data['centers']  # (N, 2, T=50) → (x, y)
del data  # Free RAM

#%% Dataset class
class EyeTrackingDataset(Dataset):
    def __init__(self, events, centers):
        self.events = events
        self.centers = centers

    def __len__(self):
        return len(self.events)
    
    def __getitem__(self, idx):
        return torch.from_numpy(self.events[idx]).float(), torch.from_numpy(self.centers[idx]).float()

    
# Split (80/20)
split = int(0.8 * len(events))
train_dataset = EyeTrackingDataset(events[:split], centers[:split])
test_dataset = EyeTrackingDataset(events[split:], centers[split:])

# DataLoaders (batch_size=32)
train_loader = DataLoader(train_dataset, batch_size=2, shuffle=True, num_workers=0)
test_loader = DataLoader(test_dataset, batch_size=2, shuffle=False, num_workers=0)

print(f"Total events: {len(events)}")
print(f"Total samples: {len(train_dataset)}")
print(f"Batch size: {train_loader.batch_size}")
print(f"Expected batches: {len(train_loader)}")
print(f"Shuffle: {train_loader.sampler.__class__.__name__}")
print(f"Num workers: {train_loader.num_workers}")

#%% ##################### 4 - Model training and evaluation ######################

def process_detector_prediction(pred):
    """Post-processing of model predictions to extract the predicted pupil coordinates for a model
    that has a centernet like head.

    Args:
        pred (torch.Tensor): shape (B, C, T, H, W)

    Returns:
        torch tensor of (B, 2) containing the x and y predicted coordinates
    """
    torch_device = pred.device
    batch_size, _, frames, height, width = pred.shape

    # Extract the center heatmap, and the x and y offset maps
    pred_pupil, pred_x_mod, pred_y_mod = pred.moveaxis(1, 0)
    pred_x_mod = torch.sigmoid(pred_x_mod)
    pred_y_mod = torch.sigmoid(pred_y_mod)

    # Find the stronger peak in the center heatmap and it's coordinates
    pupil_ind = pred_pupil.flatten(-2, -1).argmax(-1)  # (batch, frames)
    pupil_ind_x = pupil_ind % width
    pupil_ind_y = pupil_ind // width

    # Reconstruct the predicted offset
    batch_range = torch.arange(batch_size, device=torch_device).repeat_interleave(frames)
    frames_range = torch.arange(frames, device=torch_device).repeat(batch_size)
    pred_x_mod = pred_x_mod[batch_range, frames_range, pupil_ind_y.flatten(), pupil_ind_x.flatten()]
    pred_y_mod = pred_y_mod[batch_range, frames_range, pupil_ind_y.flatten(), pupil_ind_x.flatten()]

    # Express the coordinates in size agnostic terms (between 0 and 1)
    x = (pupil_ind_x + pred_x_mod.view(batch_size, frames)) / width
    y = (pupil_ind_y + pred_y_mod.view(batch_size, frames)) / height
    
    return torch.stack([x, y], dim=1)

    """ Post-processing of the model's output heatmap.

    Reconstructs the predicted x- and y- center location using numpy functions to post-process
    the output of a ONNX model.
    """
    def sigmoid(x):
        return 1 / (1 + np.exp(-x))
    
    # Squeeze time dimension: (B, C, T, H, W) → (B, C, H, W)
    # pred = pred.squeeze(2)  # Now: (batch=1, channels, height, width)
    # if pred.ndim != 4:
    #    raise ValueError(f"Expected 4D after squeeze, got {pred.shape}")

    # Pred shape is (batch, channels, height, width)
    batch_size, _, height, width = pred.shape

    # Split channels - reshape to move frames dimension after batch
    # Now (batch, height, width, channels)
    pred = np.moveaxis(pred, 1, -1)
    pred_pupil = pred[..., 0]
    pred_x_mod = sigmoid(pred[..., 1])
    pred_y_mod = sigmoid(pred[..., 2])

    # Find pupil location
    pred_pupil_flat = pred_pupil.reshape(batch_size, -1)
    pupil_ind = np.argmax(pred_pupil_flat, axis=-1)
    pupil_ind_x = pupil_ind % width
    pupil_ind_y = pupil_ind // width

    # Get the learned x- y- offset
    batch_idx = np.repeat(np.arange(batch_size)[:, None], 1, axis=1)
    x_mods = pred_x_mod[batch_idx, pupil_ind_y, pupil_ind_x]
    y_mods = pred_y_mod[batch_idx, pupil_ind_y, pupil_ind_x]

    # Calculate final coordinates
    x = (pupil_ind_x + x_mods) / width
    y = (pupil_ind_y + y_mods) / height

    return np.stack([x, y], axis=1)

import torch.optim as optim
criterion = nn.MSELoss()
optimizer = optim.AdamW(model.parameters(), lr=0.001, weight_decay=0.005)

# device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model.to('cpu')

epochs = 10

for epoch in range(epochs):
    model.train()
    train_loss = 0
    for i, (events, targets) in enumerate(train_loader):
        events, targets = events.to('cpu'), targets.to('cpu')

        optimizer.zero_grad()
        outputs = model(events)  # (B, 3, T, H, W) → heatmap + x/y offsets (B, 2, T)

        # Process outputs (use custom_process_detector_prediction)
        preds = process_detector_prediction(outputs)  # (B, 2, T) → (x, y)

        loss = criterion(preds, targets)
        loss.backward()
        optimizer.step()

        train_loss += loss.item()

        if i % 10 == 0:
            print(f"Epoch {epoch+1}/{epochs} | Batch {i} | Loss: {loss.item():.6f}")

    print(f"Epoch {epoch+1}/{epochs}, Loss: {train_loss / len(train_loader):.4f}")

    # Test loop (optional, similar but model.eval())
    # ...

# Save model
print("TRAINING DONE — MODEL SAVED")
torch.save(model.state_dict(), 'tenn_spatiotemporal_eye_trained.ckpt')



# %%
