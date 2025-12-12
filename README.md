
# Akida SNN – Event-based Pupil-Centre Detection  
**NTNU AIS4501 Specialisation Project – Event-based Vision on BrainChip Akida Neuromorphic Hardware**

Repository: https://github.com/jonurce/Akida_SNN_Event_Cameras_Eye_Center_Detection  
Author: Jon Urcelay  

## Overview
This project reproduces (from scratch) BrainChip’s official event-based pupil-centre detection example. During the projetc, the Prophesee EVK4 event camera and the Akida AKD1000 neuromorphic accelerator are used, but not used for inference with the final achieved SNN model. The complete Akida workflow is implemented and extensively documented:

1. Training of spatio-temporal CNNs (tennst backbone)  
2. INT-8 post-training quantization (default & custom calibration)  
3. CNN-to-SNN conversion  
4. Mapping to virtual (IPv2) and attempted mapping to physical (IPv1) Akida device  

Find all details on the project report `AIS4501_Specialisation_Project_Report.pdf`. A one-page poster summary is also included in `AIS4501_One_Page_Summary_Poster.pdf`.

## Repository Description

The repository is divided into three main directories:

### /EC – Event Camera ROS 2 Pipeline
Real-time event acquisition and spatio-temporal noise filtering (C++ / ROS 2 Humble Components).  
Key folder: `/EC/src/composition` – contains the custom filter (±1 px, 50 ms, min 4 events).

### /RGB – Legacy Frame-based Experiments
Initial YOLO-based object detection experiments. Kept for completeness; not used in final results.

### /AKIDA – Core Project (Pupil-Centre Detection)
All training, quantization, conversion and evaluation scripts.

#### /AKIDA/0_global_workflow
- `0_global_akida_workflow.py` – sanity-check that MetaTF and Akida environment is correctly installed.

#### /AKIDA/1_ec_example
Main working directory.

##### quantized_models/
Contains the quantized models and the akida converted SNN mdoels.

##### training/
- `event-based-eye-tracking-cvpr-2025/` – raw dataset (CVPR 2025 challenge)  
- `preprocessed_akida/` – final dataset used for Akida example replication  
- `preprocessed_akida_test_small/` – small test subset for Raspberry Pi deployment  
- `preprocessed_fast_open/` & `preprocessed_open/` – datasets for Simplified Tennst experiments  
- `runs/` – trained FP32 models (TensorBoard logs, checkpoints)  
- `plots/` – generated figures  

##### Key scripts (numbered workflow)
| Script | Purpose |
|-------|--------|
| `_0_check_cuda.py` | Verify CUDA availability |
| `_0_discover_dataset.py` | Explore raw dataset statistics |
| `_1_1_preprocess_bin_1.py` | Generate collapsed-temporal dataset (T=T=1) |
| `_1_2_preprocess_bin_10.py` | Generate T=10 dataset |
| `_1_4_preprocess_akida.py` | Generate final Akida-compatible dataset |
| `_2_3_model_f16_last_10_gradual.py` | Simplified Tennst definition |
| `_2_5_model_uint8_akida.py` | Akida Example Tennst replica |
| `_3_2_train_fast.py` / `_3_3_train_fast_dataset_10.py` / `_3_4_train_fast_akida.py` | Training scripts |
| `_4_*_test_*.py` | Evaluation & visualisation |
| `_5_0_quantize_default.py` | INT-8 quantization (default calibration) |
| `_5_1_quantize_calib.py` | INT-8 quantization with custom calibration samples |
| `_6_qat.py` | (Failed) QAT attempt |
| `_7_prep_akida_test_small.py` | Create tiny test set for Pi |
| `_8_convert.py` | CNN → Akida SNN conversion |
| `_9_test_snn.py` | Evaluate converted SNN on virtual IPv2 device |
| `_10_deploy_pi_akida.py` | Attempted deployment on real AKD1000 (fails due to IPv1/IPv2 mismatch) |

#### /AKIDA/2_custom_v1
Working directory for creating a new model compatible with Akida IPv1 (work in progress after finishing project submission).

## Dependencies
All dependencies can be found under `requirements.txt`, from which the core dependencies are:
- Python 3.10/3.11  
- PyTorch 2.x + CUDA  
- TensorFlow 2.19 + MetaTF suite (akida, quantizeml, cnn2snn)  
- Metavision SDK 4.3.0 (for event camera)  
- ROS 2 Humble (for /EC folder)
