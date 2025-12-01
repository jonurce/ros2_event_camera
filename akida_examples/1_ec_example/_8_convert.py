

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
# CONVERTION TO AKIDA SNN MODEL AND SAVE
# ============================================================

# Convert to Akida SNN model (for Akida 2.0)
try:
    akida_model = cnn2snn.convert(model_q8_onnx_int8)
    akida_model.summary()

    akida_path = AKIDA_FOLDER_PATH / "akida2_int8.fbz"
    akida_model.save(str(akida_path))
    print("Akida SNN model saved → ", akida_path)
    
except Exception as e:
    print(f"Model not fully accelerated by Akida. Reason: {str(e)}")


