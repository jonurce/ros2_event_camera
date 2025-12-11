


import pandas as pd
import matplotlib.pyplot as plt

# -------------------------------
# CHANGE THIS TO YOUR CSV PATH
# -------------------------------


simplified_T1_path = ""
simplified_T10_path = "akida_examples/1_ec_example/training/runs/tennst_5_260GB_f16_b8_e150_lr0.004/training_log_20251210_134603.csv"
akida_csv_path = "akida_examples/1_ec_example/training/runs/tennst_16_akida_b128_e150_ce_12mse_origpx/training_log_20251211_113530.csv"   # <-- put your file here

# Read CSV
df = pd.read_csv(akida_csv_path)

# Convert to numeric (in case)
df['train_l2'] = pd.to_numeric(df['train_l2'], errors='coerce')
df['val_loss_l2'] = pd.to_numeric(df['val_loss_l2'], errors='coerce')
df['epoch'] = pd.to_numeric(df['epoch'], errors='coerce')

# Validation only happens when val_loss_l2 > 0
val_mask = df['val_loss_l2'] > 0

# Plot
plt.figure(figsize=(10, 6))
plt.plot(df['epoch'], df['train_l2'], label='Train MSE Loss', color='blue', linewidth=2)
plt.plot(df['epoch'][val_mask], df['val_loss_l2'][val_mask], 
         'o-', label='Val MSE Loss', color='red', markersize=6)

plt.xlabel('Epoch')
plt.ylabel('MSE Loss (pixels)')
plt.title('Training and Validation MSE Loss in original pixel frame 640x480')
plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()

# Optional: save plot
plt.savefig('loss_akida.png', dpi=200)
plt.show()