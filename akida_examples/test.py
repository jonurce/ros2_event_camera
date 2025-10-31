
import os
os.environ['OMP_NUM_THREADS'] = '4'
os.environ['ORT_NUM_THREADS'] = '4'


import onnx
import numpy as np
from akida_models import fetch_file
onnx_checkpoint_path = "tenns_modules_onnx.onnx"
model = onnx.load(onnx_checkpoint_path)

from quantizeml.models import quantize
from quantizeml.layers import QuantizationParams

samples = fetch_file("https://data.brainchip.com/dataset-mirror/samples/eye_tracking/eye_tracking_onnx_samples_bs100.npz",
                     fname="eye_tracking_onnx_samples_bs100.npz")

# Define quantization parameters and load quantization samples
qparams = QuantizationParams(per_tensor_activations=True, input_dtype='int8')
data = np.load(samples)
# samples = np.concatenate([data[item] for item in data.files])

sample_list = list(data.files)
print(f"Found {len(sample_list)} batches. Using only first 2.")

# Take just 2 batches → ~200 MB, safe for Jetson
samples = np.concatenate([data[sample_list[i]] for i in range(min(1, len(sample_list)))])

# Quantize the model
model_quant = quantize(
    model,
    qparams=qparams,
    epochs=1,
    batch_size=1,
    samples=samples
)