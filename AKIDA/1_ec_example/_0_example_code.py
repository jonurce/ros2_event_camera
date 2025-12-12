
#%%
###################### 2 - Network architecture ######################
#%% [2.2. - Instantiating the spatiotemporal blocks]

# Show how to load and create the model
import torch
import torch.nn as nn

from tenns_modules import SpatioTemporalBlock
from torchinfo import summary

n_depthwise_layers = 4
channels = [2, 8, 16, 32, 48, 64, 80, 96, 112, 128, 256]
t_kernel_size = 5  # can vary from 1 to 10
s_kernel_size = 3  # can vary in [1, 3, 5, 7] (1 only when depthwise is False)


class TennSt(nn.Module):
    def __init__(self, channels, t_kernel_size, s_kernel_size, n_depthwise_layers):
        super().__init__()

        depthwises = [False] * (10 - n_depthwise_layers) + [True] * n_depthwise_layers
        self.backbone = nn.Sequential()
        for i in range(0, len(depthwises), 2):
            in_channels, med_channels, out_channels = channels[i], channels[i + 1], channels[i + 2]
            t_depthwise, s_depthwise = depthwises[i], depthwises[i]

            self.backbone.append(
                SpatioTemporalBlock(in_channels=in_channels, med_channels=med_channels,
                                    out_channels=out_channels, t_kernel_size=t_kernel_size,
                                    s_kernel_size=s_kernel_size, s_stride=2, bias=False,
                                    t_depthwise=t_depthwise, s_depthwise=s_depthwise))

        self.head = nn.Sequential(
            SpatioTemporalBlock(channels[-1], channels[-1], channels[-1],
                                t_kernel_size=t_kernel_size, s_kernel_size=s_kernel_size,
                                t_depthwise=False, s_depthwise=False),
            nn.Conv3d(channels[-1], 3, 1)
        )

    def forward(self, input):
        return self.head((self.backbone(input)))


model = TennSt(channels, t_kernel_size, s_kernel_size, n_depthwise_layers)
summary(model, input_size=(1, 2, 50, 96, 128), depth=4, verbose=0)
# %%
###################### 4 - Model training and evaluation ######################
#%% [4.1. - Training details]

# Load the pretrained weights in our model
from akida_models import fetch_file

ckpt_file = fetch_file(
    fname="tenn_spatiotemporal_eye.ckpt",
    origin="https://data.brainchip.com/models/AkidaV2/tenn_spatiotemporal/tenn_spatiotemporal_eye.ckpt",
    cache_subdir='models')

checkpoint = torch.load(ckpt_file, map_location="cpu")
new_state_dict = {k.replace('model._orig_mod.', ''): v for k, v in checkpoint["state_dict"].items()}
model.load_state_dict(new_state_dict)
_ = model.eval().cpu()

#%% [4.2. - Evaluation]

import numpy as np

samples = fetch_file("https://data.brainchip.com/dataset-mirror/eye_tracking_ais2024_cvpr/eye_tracking_preprocessed_400frames_test.npz",
                     fname="eye_tracking_preprocessed_400frames_test.npz")
data = np.load(samples, allow_pickle=True)
events, centers = data["events"], data["centers"]


#%% [4.2. - Evaluation]
def process_detector_prediction(pred):
    """Post-processing of model predictions to extract the predicted pupil coordinates for a model
    that has a centernet like head.

    Args:
        preds (torch.Tensor): shape (B, C, T, H, W)

    Returns:
        torch tensor of (B, 2) containing the x and y predicted coordinates
    """
    torch_device = pred.device
    batch_size, _, frames, height, width = pred.shape
    # Extract the center heatmap, and the x and y offset maps
    pred_pupil, pred_x_mod, pred_y_mod = pred.moveaxis(1, 0)
    pred_x_mod = torch.sigmoid(pred_x_mod)
    pred_y_mod = torch.sigmoid(pred_y_mod)

    # Find the stronger peak in the center heatmap and it's coordinates
    pupil_ind = pred_pupil.flatten(-2, -1).argmax(-1)  # (batch, frames)
    pupil_ind_x = pupil_ind % width
    pupil_ind_y = pupil_ind // width

    # Reconstruct the predicted offset
    batch_range = torch.arange(batch_size, device=torch_device).repeat_interleave(frames)
    frames_range = torch.arange(frames, device=torch_device).repeat(batch_size)
    pred_x_mod = pred_x_mod[batch_range, frames_range, pupil_ind_y.flatten(), pupil_ind_x.flatten()]
    pred_y_mod = pred_y_mod[batch_range, frames_range, pupil_ind_y.flatten(), pupil_ind_x.flatten()]

    # Express the coordinates in size agnostic terms (between 0 and 1)
    x = (pupil_ind_x + pred_x_mod.view(batch_size, frames)) / width
    y = (pupil_ind_y + pred_y_mod.view(batch_size, frames)) / height
    return torch.stack([x, y], dim=1)


def compute_distance(pred, center):
    """Computes the L2 distance for a prediction and center matrice

    Args:
        pred: torch tensor of shape (2, T)
        center: torch tensor of shape (2, T)
    """
    height, width = 60, 80
    pred = pred.detach().clone()
    center = center.detach().clone()
    pred[0, :] *= width
    pred[1, :] *= height
    center[0, :] *= width
    center[1, :] *= height
    l2_distances = torch.norm(center - pred, dim=0)
    return l2_distances


def pretty_print_results(collected_distances):
    """Prints the distance and accuracy within different pixel tolerance.

    By default, only the results at 20Hz will be printed (to be compatible with the
    metrics of the challenge). To print the results computed on the whole trial,
    use downsample=False. In practice, this changes very little to the final performance
    of the model.
    """
    for t in [10, 5, 3, 1]:
        p_acc = (collected_distances < t).sum() / collected_distances.size
        print(f'- p{t}: {p_acc:.3f}')
    print(f'- Euc. Dist: {collected_distances.mean():.3f} ')
#%% [4.2. - Evaluation]

# Get the model device to propagate the events properly
torch_device = next(model.parameters()).device

# Compute the distances across all 9 trials
collected_l2_distances = np.zeros((0,))
for trial_idx, event in enumerate(events):
    center = torch.from_numpy(centers[trial_idx]).float().to(torch_device)
    event = torch.from_numpy(event).unsqueeze(0).float().to(torch_device)
    pred = model(event)
    pred = process_detector_prediction(pred).squeeze(0)
    l2_distances = compute_distance(pred, center)
    collected_l2_distances = np.concatenate((collected_l2_distances, l2_distances), axis=0)

pretty_print_results(collected_l2_distances)

# %%
###################### 7 - Export to ONNX ######################

#%% [7.2 - Export to ONNX]
from tenns_modules import export_to_onnx

# Using a batch size of 10 to export with a dynamic batch size
onnx_checkpoint_path = "tenns_modules_onnx.onnx"
export_to_onnx(model, (10, 2, 50, 96, 128), out_path=onnx_checkpoint_path)


import onnx

model = onnx.load(onnx_checkpoint_path)


# %%
###################### 8 - Quantization and conversion to Akida ######################
#%% [8.1. - Quantization]
from quantizeml.models import quantize
from quantizeml.layers import QuantizationParams

# Retrieve calibration samples:
samples = fetch_file("https://data.brainchip.com/dataset-mirror/samples/eye_tracking/eye_tracking_onnx_samples_bs100.npz",
                     fname="eye_tracking_onnx_samples_bs100.npz")

# Define quantization parameters and load quantization samples
qparams = QuantizationParams(per_tensor_activations=True, input_dtype='int8')
data = np.load(samples)
samples = np.concatenate([data[item] for item in data.files])

# Quantize the model
model_quant = quantize(model, qparams=qparams, epochs=1, batch_size=100, samples=samples)

# ERROR (on Jetson):
# The Kernel crashed while executing code in the current cell or a previous cell. 
# Please review the code in the cell(s) to identify a possible cause of the failure. 
# Click <a href='https://aka.ms/vscodeJupyterKernelCrash'>here</a> for more info. 
# View Jupyter <a href='command:jupyter.viewOutput'>log</a> for further details.



#%% [8.2. - ONNX model evaluation 1]
def custom_process_detector_prediction(pred):
    """ Post-processing of the model's output heatmap.

    Reconstructs the predicted x- and y- center location using numpy functions to post-process
    the output of a ONNX model.
    """
    def sigmoid(x):
        return 1 / (1 + np.exp(-x))
    
    # Squeeze time dimension: (B, C, T, H, W) → (B, C, H, W)
    # pred = pred.squeeze(2)  # Now: (batch=1, channels, height, width)
    # if pred.ndim != 4:
    #    raise ValueError(f"Expected 4D after squeeze, got {pred.shape}")

    # Pred shape is (batch, channels, height, width)
    batch_size, _, height, width = pred.shape

    # Split channels - reshape to move frames dimension after batch
    # Now (batch, height, width, channels)
    pred = np.moveaxis(pred, 1, -1)
    pred_pupil = pred[..., 0]
    pred_x_mod = sigmoid(pred[..., 1])
    pred_y_mod = sigmoid(pred[..., 2])

    # Find pupil location
    pred_pupil_flat = pred_pupil.reshape(batch_size, -1)
    pupil_ind = np.argmax(pred_pupil_flat, axis=-1)
    pupil_ind_x = pupil_ind % width
    pupil_ind_y = pupil_ind // width

    # Get the learned x- y- offset
    batch_idx = np.repeat(np.arange(batch_size)[:, None], 1, axis=1)
    x_mods = pred_x_mod[batch_idx, pupil_ind_y, pupil_ind_x]
    y_mods = pred_y_mod[batch_idx, pupil_ind_y, pupil_ind_x]

    # Calculate final coordinates
    x = (pupil_ind_x + x_mods) / width
    y = (pupil_ind_y + y_mods) / height

    return np.stack([x, y], axis=1)

#%% [8.2. - ONNX model evaluation 2]
from onnxruntime import InferenceSession, SessionOptions
from onnxruntime_extensions import get_library_path
from quantizeml.onnx_support.quantization import ONNXModel

sess_options = SessionOptions()
sess_options.register_custom_ops_library(get_library_path())
model_quant = ONNXModel(model_quant)
session = InferenceSession(model_quant.serialized, sess_options=sess_options,
                           providers=['CPUExecutionProvider'])

#%% [8.2. - ONNX model evaluation 3]
from quantizeml.models import reset_buffers
from tqdm import tqdm

# And then evaluate the model
collected_l2_distances = []
for trial_idx, event in enumerate(events):
    center = centers[trial_idx]
    for frame_idx in tqdm(range(event.shape[1])):
        frame = event[:, frame_idx, ...][None, ...].astype(np.float32)
        pred = session.run(None, {model_quant.input[0].name: frame})[0]

        pred = custom_process_detector_prediction(pred).squeeze()
        y_pred_x = pred[0] * 80
        y_pred_y = pred[1] * 60
        center_x = center[0, frame_idx] * 80
        center_y = center[1, frame_idx] * 60
        collected_l2_distances.append(np.sqrt(np.square(
            center_x - y_pred_x) + np.square(center_y - y_pred_y)))
    # Reset FIFOs between each file
    reset_buffers(model_quant)
#%% [8.2. - ONNX model evaluation 4]
pretty_print_results(np.array(collected_l2_distances))
#%% [8.3. - Conversion to Akida]
from cnn2snn import convert

akida_model = convert(model_quant.model)
akida_model.summary()
# %% [8.4. - Check performance]
# How to do this?????
# accuracy = akida_model.evaluate(x_test, y_test.astype(np.int32))
# print('Test accuracy after conversion:', accuracy)
