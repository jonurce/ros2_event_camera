# %%
###################### 7 - FIFO buffering for streaming inference ######################
#%% [7.2. - Exporting to ONNX]
from tenns_modules import export_to_onnx

# Using a batch size of 10 to export with a dynamic batch size
onnx_checkpoint_path = "tenns_modules_onnx.onnx"
export_to_onnx(model, (10, 2, 50, 96, 128), out_path=onnx_checkpoint_path)

#%% [7.2. - Importing from ONNX]
import onnx

model = onnx.load(onnx_checkpoint_path)