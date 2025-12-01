

import sys
from pathlib import Path
import os
import tensorflow as tf
import onnx

# Disable GPU for TF (quantizeml) -> FORCE quantizeml TO RUN ON CPU
os.environ["CUDA_VISIBLE_DEVICES"] = ""          # ← Disable GPU completely for TF/quantizeml
tf.config.set_visible_devices([], 'GPU')         # ← Extra safety — hide GPU from TF


import cnn2snn






# ============================================================
# CONFIG 
# ============================================================

# Input / output sizes
W, H = 640, 480              # Original input size (pixels)
W_IN, H_IN = 128, 96         # Model input size (pixels)
W_OUT, H_OUT = 4, 3          # Output heatmap size (out coordinates)


QUANTIZED_FOLDER_PATH = Path("akida_examples/1_ec_example/quantized_models/q8_calib_b8_n10")
Q_INT8_PATH = QUANTIZED_FOLDER_PATH / "q_tennst_int8.onnx"

AKIDA_FOLDER_PATH = QUANTIZED_FOLDER_PATH / "akida_models"
AKIDA_FOLDER_PATH.mkdir(exist_ok=True)







# ============================================================
# LOAD & VERIFY SAVED QUANTIZED INT8 MODEL
# ============================================================

print("\nVerifying saved INT8 model structure...")
model_q8_onnx_int8 = onnx.load(str(Q_INT8_PATH))
print("Int8 ONNX model loaded")
print("Input shape:", [x.dim_value or x.dim_param for x in model_q8_onnx_int8.graph.input[0].type.tensor_type.shape.dim])
print("Output shape:", [x.dim_value or x.dim_param for x in model_q8_onnx_int8.graph.output[0].type.tensor_type.shape.dim])









# ============================================================
# CHECK COMPATIBILITY FOR AKIDA 2.0
# ============================================================

print("Checking cnn2snn Akida Version")
current_version = cnn2snn.get_akida_version()
print(f'Current version: {current_version}')
cnn2snn.set_akida_version(cnn2snn.AkidaVersion.v2)
updated_version = cnn2snn.get_akida_version()
print(f'Current version: {updated_version}')

print("Checking model compatibility for Akida 2.0...")
# compatibility = cnn2snn.check_model_compatibility(Q_INT8_PATH, target_akida_version=2)
compatibility = cnn2snn.check_model_compatibility(model_q8_onnx_int8)
if compatibility['overall'] != 'compatible':
    print("WARNING: Model has issues. Fix quantization/conversion params and retry.")
    print(compatibility)  # Shows details
    exit(1)
else:
    print("Model is compatible with Akida 2.0!")










# ============================================================
# CONVERTION TO AKIDA 2.0 SNN MODEL AND SAVE
# ============================================================

print("Converting to Akida 2.0...")
# Convert to Akida SNN model (for Akida 2.0)
try:
    # akida_model = cnn2snn.convert(model_q8_onnx_int8, target_akida_version=2)
    akida_model = cnn2snn.convert(model_q8_onnx_int8)
    akida_model.summary()

    akida_path = AKIDA_FOLDER_PATH / "akida2_int8_v2.fbz"
    akida_model.save(str(akida_path))
    print("Akida SNN model saved → ", akida_path)
    
except Exception as e:
    print(f"Model not fully accelerated by Akida. Reason: {str(e)}")


