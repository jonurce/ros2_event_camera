# _3_train_v1_compatible.py
# Training script for v1-compatible pupil centre detector (no temporal dimension)
# Input: [B, 2, 96, 128] float32 normalized (pos/neg channels)
# Target: [B, 3] → (confidence=1.0, x_norm [0,1], y_norm [0,1])
# Loss: BCE on confidence + MSE on coordinates (weighted)

import tensorflow as tf
import torch
import tf_keras
import numpy as np
from pathlib import Path
from datetime import datetime
import csv
from tqdm import tqdm
import os


# ================================================
# MULTI-GPU SETUP (MirroredStrategy)
# ================================================
gpus = tf.config.list_physical_devices('GPU')
if len(gpus) > 1:
    print(f"Found {len(gpus)} GPUs — using MirroredStrategy for multi-GPU training")
    strategy = tf.distribute.MirroredStrategy()  # Uses all available GPUs
else:
    print("Only 1 or 0 GPUs found — falling back to default strategy (single GPU or CPU)")
    strategy = tf.distribute.get_strategy()  # Default = single GPU or CPU

print(f"Number of devices: {strategy.num_replicas_in_sync}")


# ================================================
# CONFIG
# ================================================
BATCH_SIZE_PER_REPLICA = 64
GLOBAL_BATCH_SIZE = BATCH_SIZE_PER_REPLICA * strategy.num_replicas_in_sync  # Scales with GPUs

NUM_EPOCHS = 150
LEARNING_RATE = 0.002
WEIGHT_DECAY = 0.005

DATA_ROOT = Path("/home/dronelab-pc-1/Jon/IndustrialProject/AKIDA/2_custom_v1/preprocessed_v1")
LOG_DIR = Path(f"/home/dronelab-pc-1/Jon/IndustrialProject/AKIDA/2_custom_v1/runs/v1_tennst_b{BATCH_SIZE_PER_REPLICA}_e{NUM_EPOCHS}")
LOG_DIR.mkdir(parents=True, exist_ok=True)
LOG_FILE = LOG_DIR / f"training_log_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"

W_ORIG, H_ORIG = 640, 480  # Original frame size (for pixel conversion in metrics)

print(f"Training v1-compatible model | Batch: {BATCH_SIZE_PER_REPLICA} | Epochs: {NUM_EPOCHS}")


# ================================================
# DATASET — loads normalized voxels + normalized heatmaps
# ================================================
class V1GazeDataset(tf.keras.utils.Sequence):
    def __init__(self, split="train", batch_size=BATCH_SIZE_PER_REPLICA, data_root=DATA_ROOT):
        self.batch_size = batch_size
        self.split = split
        self.samples = []

        split_path = data_root / split
        for rec_dir in sorted(split_path.iterdir()):
            if not rec_dir.is_dir():
                continue

            voxels_path = rec_dir / "voxels.pt"
            heatmaps_path = rec_dir / "heatmaps.pt"

            if not voxels_path.exists() or not heatmaps_path.exists():
                continue

            # Memory-map → zero RAM usage until accessed
            # voxels = torch.load(voxels_path, map_location="cpu", mmap=True, weights_only=True)
            heatmaps = torch.load(heatmaps_path, map_location="cpu", mmap=True, weights_only=True)

            N = len(heatmaps)

            for i in range(N):
                self.samples.append((rec_dir, i))

        print(f"{split.upper()} → {len(self.samples)} samples")

    def __len__(self):
        return (len(self.samples) + self.batch_size - 1) // self.batch_size

    def __getitem__(self, idx):
        rec_dir, i = self.samples[idx]

        # [2,96,128] float32
        voxels = torch.load(rec_dir / "voxels.pt", map_location="cpu", mmap=True, weights_only=True)[i]  
            
        # [3] float32   
        heatmap = torch.load(rec_dir / "heatmaps.pt", map_location="cpu", mmap=True, weights_only=True)[i]  

        return voxels.numpy(), heatmap.numpy()  # [B, 2, 96, 128], [B, 3]


# ================================================
# MODEL DEFINITION — v1-compatible (from your fixed code)
# ================================================
with strategy.scope():
    def create_v1_tennst(input_shape=(96, 128, 2)):
        model = tf_keras.Sequential(name="v1_compatible_tennst")

        model.add(tf_keras.layers.Input(shape=input_shape))
        model.add(tf_keras.layers.Conv2D(filters=8, kernel_size=1, strides=1, padding='same', use_bias=False))
        model.add(tf_keras.layers.ReLU(max_value=6.0))

        model.add(tf_keras.layers.SeparableConv2D(filters=16, kernel_size=3, strides=2, padding='same', use_bias=False))
        model.add(tf_keras.layers.ReLU(max_value=6.0))
        model.add(tf_keras.layers.MaxPooling2D(pool_size=2, padding='same'))

        model.add(tf_keras.layers.SeparableConv2D(filters=32, kernel_size=3, strides=2, padding='same', use_bias=False))
        model.add(tf_keras.layers.ReLU(max_value=6.0))
        model.add(tf_keras.layers.MaxPooling2D(pool_size=2, padding='same'))

        model.add(tf_keras.layers.SeparableConv2D(filters=48, kernel_size=3, strides=2, padding='same', use_bias=False))
        model.add(tf_keras.layers.ReLU(max_value=6.0))
        model.add(tf_keras.layers.MaxPooling2D(pool_size=2, padding='same'))

        model.add(tf_keras.layers.SeparableConv2D(filters=80, kernel_size=3, strides=2, padding='same', use_bias=False))
        model.add(tf_keras.layers.ReLU(max_value=6.0))

        model.add(tf_keras.layers.SeparableConv2D(filters=112, kernel_size=3, strides=2, padding='same', use_bias=False))
        model.add(tf_keras.layers.ReLU(max_value=6.0))

        model.add(tf_keras.layers.Conv2D(filters=256, kernel_size=1, use_bias=False))
        model.add(tf_keras.layers.ReLU(max_value=6.0))
        model.add(tf_keras.layers.Conv2D(filters=3, kernel_size=1, activation=None))  # [1,1,3]

        return model
    
    model = create_v1_tennst()

    def v1_loss(true, pred):

        # true: [B, 3] → confidence, x_norm, y_norm
        # pred: [B, 1, 1, 3] → confidence, x_norm, y_norm
        # Squeeze spatial dims: [B, 1, 1, 3] → [B, 3]
        pred = tf.squeeze(pred, axis=[1, 2])  # [B, 3]

        conf_true = true[:, 0]   # always 1.0
        conf_pred = pred[:, 0]

        x_true = true[:, 1]
        y_true = true[:, 2]

        x_pred = pred[:, 1]
        y_pred = pred[:, 2]

        # BCE on confidence (encourage high confidence)
        loss_conf = tf_keras.losses.binary_crossentropy(tf.ones_like(conf_true), tf.sigmoid(conf_pred))

        # MSE on normalized coordinates
        loss_coord = tf_keras.losses.mse(tf.stack([x_true, y_true], axis=1),
                                        tf.stack([tf.sigmoid(x_pred), tf.sigmoid(y_pred)], axis=1))

        # Weighted sum (confidence less important)
        return loss_conf + 12.0 * loss_coord

    def pixel_error(true, pred):

        # true: [B, 3] → confidence, x_norm, y_norm
        # pred: [B, 1, 1, 3] → confidence, x_norm, y_norm
        # Squeeze spatial dims: [B, 1, 1, 3] → [B, 3]
        pred = tf.squeeze(pred, axis=[1, 2])  # [B, 3]

        x_true = true[:, 1] * W_ORIG
        y_true = true[:, 2] * H_ORIG

        x_pred = tf.sigmoid(pred[:, 1]) * W_ORIG
        y_pred = tf.sigmoid(pred[:, 2]) * H_ORIG

        return tf.sqrt((x_pred - x_true)**2 + (y_pred - y_true)**2)
    
    model.compile(
        optimizer=tf_keras.optimizers.AdamW(learning_rate=LEARNING_RATE, weight_decay=WEIGHT_DECAY),
        loss=v1_loss,
        metrics=[pixel_error]
    )

# ================================================
# DATA LOADING (outside scope — tf.data handles distribution)
# ================================================
train_ds = V1GazeDataset("train")
val_ds = V1GazeDataset("test")

# NCHW → NHWC: [N, 96, 128, 2] float32
train_gen = tf.data.Dataset.from_generator(
    lambda: train_ds,
    output_signature=(
        tf.TensorSpec(shape=(2, 96, 128), dtype=tf.float32),
        tf.TensorSpec(shape=(3), dtype=tf.float32)
    )
).map(lambda x, y: (tf.transpose(x, [1, 2, 0]), y)).batch(GLOBAL_BATCH_SIZE).prefetch(tf.data.AUTOTUNE)

val_gen = tf.data.Dataset.from_generator(
    lambda: val_ds,
    output_signature=(
        tf.TensorSpec(shape=(2, 96, 128), dtype=tf.float32),
        tf.TensorSpec(shape=(3), dtype=tf.float32)
    )
).map(lambda x, y: (tf.transpose(x, [1, 2, 0]), y)).batch(GLOBAL_BATCH_SIZE).prefetch(tf.data.AUTOTUNE)


# ================================================
# CALLBACKS & TRAINING
# ================================================
callbacks_list = [
    tf_keras.callbacks.CSVLogger(str(LOG_FILE)),
    tf_keras.callbacks.ReduceLROnPlateau(monitor='val_pixel_error', factor=0.5, patience=10),
    tf_keras.callbacks.ModelCheckpoint(str(LOG_DIR / "best.h5"), save_best_only=True, monitor='val_pixel_error')
]

model.fit(
    train_gen,
    epochs=NUM_EPOCHS,
    validation_data=val_gen,
    callbacks=callbacks_list
)

model.save(str(LOG_DIR / "final.h5"))
print("Training complete! Best model saved in:", LOG_DIR)