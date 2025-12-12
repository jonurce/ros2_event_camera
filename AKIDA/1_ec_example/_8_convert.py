

import sys
from pathlib import Path
import os
import tensorflow as tf
import onnx

# Disable GPU for TF (quantizeml) -> FORCE quantizeml TO RUN ON CPU
os.environ["CUDA_VISIBLE_DEVICES"] = ""          # ← Disable GPU completely for TF/quantizeml
tf.config.set_visible_devices([], 'GPU')         # ← Extra safety — hide GPU from TF


import cnn2snn
from cnn2snn import get_akida_version, set_akida_version, AkidaVersion






# ============================================================
# CONFIG 
# ============================================================

# Input / output sizes
W, H = 640, 480              # Original input size (pixels)
W_IN, H_IN = 128, 96         # Model input size (pixels)
W_OUT, H_OUT = 4, 3          # Output heatmap size (out coordinates)


QUANTIZED_FOLDER_PATH = Path("akida_examples/1_ec_example/quantized_models/q8_calib_GOOD_b8_n10")
Q_INT8_PATH = QUANTIZED_FOLDER_PATH / "q_tennst_int8.onnx"
TENNST_PATH = QUANTIZED_FOLDER_PATH / "tennst.onnx"

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

# current_version = cnn2snn.get_akida_version()
# print(f'Current version: {current_version}')

# cnn2snn.set_akida_version(cnn2snn.AkidaVersion.v1)
# cnn2snn.set_akida_version(version=cnn2snn.AkidaVersion.v1)
target_version=AkidaVersion.v1
set_akida_version(target_version)
updated_version = get_akida_version()
print(f'Target version: {target_version} \n')
print(f'Updated version: {updated_version} \n')





# print("Checking model compatibility for Akida 2.0...")
# model_tennst_onnx = onnx.load(str(TENNST_PATH))

# from akida import devices 
# detected_device = devices()[0]

# compatibility = cnn2snn.check_model_compatibility(model_q8_onnx_int8, device=detected_device, input_dtype="uint8")
# compatibility = cnn2snn.check_model_compatibility(model_q8_onnx_int8)

# if compatibility['overall'] != 'compatible':
#     print("WARNING: Model has issues. Fix quantization/conversion params and retry.")
#     print(compatibility)  # Shows details
#     exit(1)
# else:
#     print("Model is compatible with Akida 2.0!")










# ============================================================
# CONVERTION TO AKIDA 2.0 SNN MODEL AND SAVE
# ============================================================

print("Converting to Akida 2.0...")
# Convert to Akida SNN model (for Akida 2.0)
try:
    akida_model = cnn2snn.convert(model_q8_onnx_int8)
    akida_model.summary()

    akida_path = AKIDA_FOLDER_PATH / "akida_int8_v2.fbz"
    akida_model.save(str(akida_path))
    print("Akida SNN model saved → ", akida_path)
    
except Exception as e:
    print(f"Model not fully accelerated by Akida. Reason: {str(e)}")



# ============================================================
# MAP TO AKIDA HARDWARE NSOC
# ============================================================
import akida 
devices = akida.devices()
print(f'Available devices: {[dev.desc for dev in devices]}')
assert len(devices), "No device found, this example needs an Akida NSoC_v2 device."
device = devices[0]
assert device.version == akida.NSoC_v2, "Wrong device found, this example needs an Akida NSoC_v2."
print(f"Found Akida device: {device}")
print(f"Akida device IP version: {device.ip_version}")
print(f"Akida device version: {device.version}\n")

print(f"Akida HwVersion IP version: {akida.HwVersion.ip_version}\n")

print(f"Akida model device: {akida_model.device}")
print(f"Akida model IP version: {akida_model.ip_version}")
print(f"Akida model MACs: {akida_model.macs}\n")

print(f"Akida module version: {akida.__version__}")

# Create v2 virtual device (emulates SixNodesIPv2)
device_v2 = akida.SixNodesIPv2()

print(f"Virtual device IP: {device_v2.ip_version}")  # IpVersion.v2


akida_model.map(device_v2)     # important! map model to NSOC
print("Model mapped to hardware — ready for ultra-low power inference")

# Check model mapping: NP allocation and binary size
akida_model.summary()
