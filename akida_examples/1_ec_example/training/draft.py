
# Check pytorch:
import torch
import numpy as np

print("=== JETSON CUDA CHECK ===")
print("PyTorch:", torch.__version__)
print("CUDA available:", torch.cuda.is_available())
print("CUDA version:", torch.version.cuda)
print("GPU:", torch.cuda.get_device_name(0) if torch.cuda.is_available() else "N/A")
print("cuBLAS enabled:", torch.backends.cudnn.enabled)

print("\n--- Testing matmul ---")
a = torch.randn(128, 128).cpu()
b = torch.randn(128, 128).cpu()
c = a @ b
print("MATMUL OK:", c.sum().item())


# Check tenns_modules: hi
from tenns_modules import SpatioTemporalBlock
print('\nSuccess with tenns_modules!')