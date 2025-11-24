# 2_5_model_unit8_akida.py

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
        channels = [2, 8, 16, 32, 48, 64, 80, 96, 112, 128, 256] # original
        # channels = [1, 8, 16, 32, 48, 64, 80, 96, 112, 128, 256]  # starts with 1!
        

        # Depthwise in last N layers → Akida-friendly
        depthwises = [False] * (10 - n_depthwise_layers) + [True] * n_depthwise_layers

        # [1, 2, 50, 96, 128] -> dw=False [1, 16, 50, 48, 64] -> dw=False [1, 48, 50, 24, 32] ->
        # -> dw=False [1, 80, 50, 12, 16] -> dw=True [1, 112, 50, 6, 8] -> dw=True [1, 256, 50, 3, 4]
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
                                    bias = False,
                                    t_depthwise=t_dw,
                                    s_depthwise=s_dw)
            )

        # Final feature extractor [1, 256, 50, 3, 4] → [1, 3, 50, 3, 4]
        self.head = nn.Sequential(
            SpatioTemporalBlock(
                in_channels=channels[-1],
                med_channels=channels[-1],
                out_channels=channels[-1],
                t_kernel_size=t_kernel_size,
                s_kernel_size=s_kernel_size,
                t_depthwise=False,
                s_depthwise=False
            ),
            nn.Conv3d(channels[-1], 3, 1)
        )
  
    def forward(self, input):
        """
        x: [B, 2, 50, 96, 128] int8
        """
        input = input.float()  # → [B, 2, 50, 96, 128]

        # Backbone: spatio-temporal feature extraction
        # [B, 2, 50, 96, 128] -> [B, 256, 50, 3, 4]
        input = self.backbone(input)
        # [B, 256, 50, 3, 4] -> [B, 3, 50, 3, 4]
        input = self.head(input)   
        return input


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
    # if torch.cuda.device_count() > 1:
    #     print(f"Using {torch.cuda.device_count()} GPUs with DataParallel...")
    #     model = torch.nn.DataParallel(model)

    # TORCH.COMPILE — still great, now even more stable
    # print("Compiling model with torch.compile()...")
    # model = torch.compile(model, mode="reduce-overhead")

    # model = model.float()

    model.eval()

    # Simulate one batch of your data B=4
    dummy_input = torch.randn((4, 2, 50, 96, 128), dtype=torch.uint8)

    with torch.no_grad():
        pred = model(dummy_input)

    print(f"Input:  {dummy_input.shape} (uint8)")
    print(f"Prediction:   {pred.shape}")

    # Model summary
    summary(model, input_size=(1, 2, 50, 96, 128), device=DEVICE)