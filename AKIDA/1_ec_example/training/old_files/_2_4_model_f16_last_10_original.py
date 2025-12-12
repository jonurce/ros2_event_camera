# 2_model.py
# TennSt for Eye-Tracking: predicts (x, y) + eye state (open/closed)
# Input:  [B, 10, 640, 480] int8 → 10 time bins of event polarity sum
# Output: [B, 2] gaze + [B, 1] state probability

import torch
import torch.nn as nn
from tenns_modules import SpatioTemporalBlock
from torchinfo import summary

# ===================================================================
# Our TennSt Model — 10 time bins → (x,y) + state
# ===================================================================
class EyeTennSt(nn.Module):
    def __init__(self,
                 t_kernel_size=5,
                 s_kernel_size=3,
                 n_depthwise_layers=4):
        super().__init__()

        # Channel progression — tuned for 640×480 → fast downsampling
        # channels = [2, 8, 16, 32, 48, 64, 80, 96, 112, 128, 256] # original
        channels = [1, 8, 16, 32, 48, 64, 80, 96, 112, 128, 256]  # starts with 1!
        

        # Depthwise in last N layers → Akida-friendly
        depthwises = [False] * (10 - n_depthwise_layers) + [True] * n_depthwise_layers

        # [1, 1, 10, 640, 480] -> dw=False [1, 16, 10, 320, 320] -> dw=False [1, 48, 10, 160, 160] ->
        # -> dw=False [1, 96, 10, 80, 80] -> dw=True [1, 160, 10, 40, 40] -> dw=True [1, 256, 10, 20, 20]
        self.backbone = nn.Sequential()
        for i in range(0, len(depthwises), 2): # 0, 2, 4, 6, 8
            in_c, med_c, out_c = channels[i], channels[i + 1], channels[i + 2]
            t_dw, s_dw  = depthwises[i], depthwises[i]

            self.backbone.append(
                SpatioTemporalBlock(in_channels=in_c,
                                    med_channels=med_c,
                                    out_channels=out_c,
                                    t_kernel_size=t_kernel_size,
                                    s_kernel_size=s_kernel_size,
                                    s_stride=2,     # aggressive downsampling
                                    # bias = False
                                    t_depthwise=t_dw,
                                    s_depthwise=s_dw)
            )

        # Final feature extractor [1, 256, 10, 20, 20] → [1, 256, 10, 20, 20]
        self.pre_head = SpatioTemporalBlock(
            in_channels=channels[-1],
            med_channels=channels[-1],
            out_channels=channels[-1],
            t_kernel_size=t_kernel_size,
            s_kernel_size=s_kernel_size,
            t_depthwise=False,
            s_depthwise=False
        )
  
        # Two heads: gaze and state
        # [1, 256, 1, 1, 1] → [1, 2]
        self.gaze_head  = nn.Conv3d(channels[-1], 2, 1)  # (x, y)
        # [1, 256, 1, 1, 1] → [1, 1]
        self.state_head = nn.Conv3d(channels[-1], 1, 1)  # binary logit

    def forward(self, input):
        """
        x: [B, 10, 640, 480] float 16
        """
        # Add channel dimension: [B, 10, 640, 480] → [B, 1, 10, 640, 480]
        if input.dim() == 4:
            input = input.unsqueeze(1).float()  # → [B, 1, T, H, W]

        # Backbone: spatio-temporal feature extraction
        # [B, 1, 10, 640, 480] -> [B, 256, 10, 20, 20]
        input = self.backbone(input)
        # [B, 256, 10, 20, 20] -> [B, 256, 10, 20, 20]
        input = self.pre_head(input)        

        # Predictions
        # [B, 256, 10, 20, 20] -> [B, 2, 1, 1, 1]
        gaze  = self.gaze_head(input) 
        # [B, 256, 10, 20, 20] -> [B, 2, 1, 1, 1]  
        state = self.state_head(input) 

        return gaze, state


# ===================================================================
# Quick test
# ===================================================================
DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
torch.cuda.empty_cache()
if __name__ == "__main__":
    model = EyeTennSt(
        t_kernel_size=5,
        s_kernel_size=3,
        n_depthwise_layers=4
    ).to(DEVICE)

    # DATA PARALLEL for multi-GPU setups
    if torch.cuda.device_count() > 1:
        print(f"Using {torch.cuda.device_count()} GPUs with DataParallel...")
        model = torch.nn.DataParallel(model)

    # TORCH.COMPILE — still great, now even more stable
    print("Compiling model with torch.compile()...")
    model = torch.compile(model, mode="reduce-overhead")

    model = model.float()

    # Simulate one batch of your data B=4
    # dummy_input = torch.randn((4, 10, 640, 480), dtype=torch.float16)
    dummy_input = torch.randn((4, 10, 80, 60), dtype=torch.float16)

    gaze_pred, state_pred = model(dummy_input)
    print(f"Input:  {dummy_input.shape} (float)")
    print(f"Gaze:   {gaze_pred.shape}")
    print(f"State:  {state_pred.shape}")

    # Model summary
    summary(model, input_size=(1, 10, 640, 480), device="cuda")