
import time
import torch
from pathlib import Path
import akida


# ============================================================
# LOAD SAVED SNN IPV! TEST MODEL
# ============================================================


AKIDA_FOLDER_PATH = Path("AKIDA/1_ec_example/quantized_models/q8_calib_GOOD_b8_n10/akida_V1_models")
AKIDA_PATH = AKIDA_FOLDER_PATH / "akida_v1.fbz"
akida_model_q4 = akida.Model(str(AKIDA_PATH)) 
print("\n ----------- LOADED AKIDA SNN MODEL -----------")
print(f"Akida model device: {akida_model_q4.device}")
print(f"Akida model IP version: {akida_model_q4.ip_version}")
print(f"Akida model MACs: {akida_model_q4.macs}\n")
print("Model summary:")
akida_model_q4.summary()




# ============================================================
# TEST MODEL MAPPED INTO VIRTUAL DEVICE WITH RANDOM INFERENCE
# ============================================================

# Input shape for test model: [B=1, 28, 28, 1]
dummy_input = torch.randint(0, 30, (1, 28, 28, 1), dtype=torch.uint8)
x = dummy_input.numpy() 

start_virtual = time.time()
pred = akida_model_q4.predict(x)   
latency_virtual_ms = (time.time() - start_virtual) * 1000

print("\nSuccessful inference into model mapped into virtual AKD1000.")
print(f"Latency (ms): {latency_virtual_ms}")






# ============================================================
# MAP TO AKIDA HARDWARE NSOC
# ============================================================

devices = akida.devices()
print(f'\nAvailable devices: {[dev.desc for dev in devices]}')
assert len(devices), "No device found, this example needs an Akida NSoC_v2 device."

device = devices[0]
assert device.version == akida.NSoC_v2, "Wrong device found, this example needs an Akida NSoC_v2."
print(f"\nFound Akida device: {device}")
print(f"Akida device IP version: {device.ip_version}")
print(f"Akida device version: {device.version}\n")

akida_model_q4.map(device)  
print("Model mapped into AKD1000 in the Pi!")
print("Mapped model summary:")
akida_model_q4.summary()









# ============================================================
# TEST MODEL MAPPED INTO REAL DEVICE WITH RANDOM INFERENCE
# ============================================================

start_pi = time.time()
pred = akida_model_q4.predict(x)   
latency_pi_ms = (time.time() - start_pi) * 1000

print("\nSuccessful inference into model mapped into Pi AKD1000.")
print(f"Latency (ms): {latency_pi_ms}")



