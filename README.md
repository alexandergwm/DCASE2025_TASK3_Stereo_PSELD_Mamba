# DCASE2025_TASK3_Stereo_PSELD_Mamba

## Table of contents

- [DCASE2025_TASK3_Stereo_PSELD_Mamba](#dcase2025_task3_stereo_pseld_mamba)
  - [Table of contents](#table-of-contents)
  - [Introduction](#introduction)
  - [Environments](#environments)
  - [Project Structure](#project-structure)
  - [Dataset](#dataset)
    - [DCASE 2025 Task 3 Dataset](#dcase-2025-task-3-dataset)
  - [Download model checkpoints](#download-model-checkpoints)
    - [1. Download PSELDNets checkpoints](#1-download-pseldnets-checkpoints)
    - [2. Download Stereo-enhanced SELD pretrained models](#2-download-stereo-enhanced-seld-pretrained-models)
  - [Quick Start](#quick-start)
    - [Complete Workflow](#complete-workflow)
    - [Feature Extraction](#feature-extraction)
    - [Inference](#inference)
    - [Training](#training)
    - [Model Analysis](#model-analysis)
  - [Cite](#cite)
  - [References](#references)


## Introduction

This repo contains code for our DCASE 2025 task3 proposed method : **Stereo sound event localization and detection based on PSELDnet pretraining and BiMamba sequence modeling** [1]. For more information, please read the paper [here](https://arxiv.org/abs/2506.13455).

The features of this method are:

* It introduces PSELDNets trained on large-scale synthetic SELD datasets to enhance the performance of Stereo tasks in DoA, SEC, and DDOA tasks.
* It replaces Conformer with BiMamba architecture and introduces asymmetric convolution modules to improve performance.


## Environments

### System Requirements


The codebase is developed with Python 3.11.11. Install requirements are as follows:

```bash
# install full dependencies with optional packages
pip install -r requirements.txt
```

## Project Structure

```
DCASE_2025_task3/2025_my_pretrained/
├── main.py                          # Main training script
├── inference.py                     # Inference script
├── evaluate.py                      # Evaluation script
├── run_experiment.py                # Experiment runner
├── model.py                         # Model architectures
├── utils.py                         # Utility functions
├── metrics.py                       # Evaluation metrics
├── loss.py                          # Loss functions
├── data_generator_aug.py            # Data loading with augmentation
├── extract_features_aug.py          # Feature extraction
├── baseline_config.py               # Base configuration
├── requirements.txt                 # Full dependencies
├── experiments/                     # Experiment configurations
│   ├── baseline_config.py
│   ├── EXP_HTSAT.py
│   ├── EXP_CNN14_Conformer.py
│   ├── EXP_CNN14_BiMamba.py
│   ├── EXP_CNN14_BiMambaAC.py
│   └── ...
├── pretrained_model/                # Pretrained model components
├── PSELD_pretrained_ckpts/          # PSELDNets pretrained checkpoints
├── pretrained_ckpts/                # Stereo-enhanced SELD pretrained models
├── features/                        # Extracted features
├── outputs/                         # Model outputs
├── checkpoints/                     # Training checkpoints
└── logs/                            # Training logs
```

## Dataset

### DCASE 2025 Task 3 Dataset

The DCASE 2025 Task 3 dataset can be downloaded from [Zenodo](https://zenodo.org/records/15559774).

**Setup Instructions:**
1. Download the dataset from the link above
2. Extract the dataset to your desired location
3. Update the dataset path in `/experiments/baseline_config.py`:
   ```python
   # Find the dataset configuration section and modify:
   'root_dir': '/path/to/your/DCASE2025_DATASET',
   'feature_root_dir': '/path/to/your/DCASE2025_DATASET/features',
   ```
   
**Important Notes:**
- Replace `/path/to/your/DCASE2025_DATASET` with your actual dataset path
- Make sure the path uses forward slashes (/) even on Windows
- The features directory will be created automatically during feature extraction

**Dataset Structure:**
```
DCASE2025_DATASET/
├── audio/
│   ├── dev/
│   │   ├── fold1/
│   │   ├── fold2/
│   │   ├── fold3/
│   │   └── fold4/
│   └── eval/
└── metadata/
    ├── dev/
    └── eval/
```

**Dataset Details:**
- **Audio Format**: Stereo, 24 kHz sampling rate
- **Duration**: 5-second audio clips  
- **Classes**: 13 sound event classes
- **Tasks**: Sound Event Localization and Detection (SELD)

## Download model checkpoints

### 1. Download PSELDNets checkpoints

Download PSELDNets corresponding checkpoints from [Google Drive](https://drive.google.com/drive/folders/1GHw-_NjW9UNyJ1rpxCpZ985SOb2RVyPA?usp=sharing) or [HuggingFace](https://huggingface.co/datasets/Jinbo-HU/PSELDNets). This work only uses the pretrained models of HTSAT and CNN14-Conformer networks. Place them in the `PSELD_pretrained_ckpts` folder, More pre-trained checkpoints of PSELDNets can be found in [PSELDNets](https://github.com/Jinbo-Hu/PSELDNets). 

```bash
# Create the directory
mkdir -p PSELD_pretrained_ckpts

# Download and place the following checkpoints:
# - mACCDOA-HTSAT-0.567.ckpt
# - mACCDOA-CNN14-Conformer-0.582.ckpt
```

### 2. Download Stereo-enhanced SELD pretrained models

Download the pretrained models for the Stereo-enhanced SELD method from OneDrive [[link](https://1drv.ms/f/c/8129c8ae660ec19d/ElIqb46onPxDkps-BWoP-XABK5qPj_sjlV6eqwYnxXEERw?e=Il6SGs)] and place them in the `pretrained_ckpts` folder.

```bash
# Create the directory
mkdir -p pretrained_ckpts

# Download the Stereo-enhanced SELD pretrained models from OneDrive
# Place all downloaded models in this folder
# Available models:
# - CNN14_BiMamba
# - CNN14_BiMambaAC  
# - CNN14_Conformer
# - CNN14_ConBimamba
# - CNN14_ConBiMambaAC
# - HTSAT
```

## Quick Start

### Complete Workflow

**Step 1: Environment Setup**
```bash
# Install dependencies
pip install -r requirements.txt
```

**Step 2: Dataset and Configuration**
1. Download DCASE 2025 Task 3 dataset (see Dataset section)
2. Configure paths in `baseline_config.py`

**Step 3: Feature Extraction**
```bash
python extract_features_aug.py
```

**Step 4: Choose your approach**
- **For inference**: Use pretrained models (see Inference section)
- **For training**: Train your own models (see Training section)

---

### Feature Extraction

**Before running feature extraction:**
1. Set the original dataset file path in `baseline_config.py`:
   ```python
   'root_dir': '/path/to/your/DCASE2025_DATASET',
   'feature_dir': '/path/to/your/features'
   ```

2. Extract PFOA (Pseudo First-Order Ambisonics) features from audio:
   ```bash
   python extract_features_aug.py
   ```
   
   **Note**: Feature extraction may take several minutes depending on:
   - Dataset size (full DCASE 2025 Task 3 dataset)
   - System performance (CPU/GPU specifications)
   - Storage speed (SSD recommended for faster I/O)

**Feature Details:**
- **PFOA Features**: 7-channel input (4 channels Log-Mel + 3 channels Intensity Vectors)
- **Feature Type**: Pseudo First-Order Ambisonics with intensity vectors
- **Augmentation**: LRswap (Left-Right channel swapping) for data augmentation
- **Output**: Preprocessed features saved for training and inference



### Inference
Get the results in paper directly via inference on the DCASE 2025 Task 3 dataset using our pretrained models:

```bash
# Inference using CNN14-BiMamba model
python inference.py --exp EXP_CNN14_BiMamba --checkpoints_dir pretrained_ckpts/CNN14_BiMamba

# Inference using CNN14-BiMambaAC model  
python inference.py --exp EXP_CNN14_BiMambaAC --checkpoints_dir pretrained_ckpts/CNN14_BiMambaAC

# Inference using CNN14-ConBimamba model
python inference.py --exp EXP_CNN14_ConBimamba --checkpoints_dir pretrained_ckpts/CNN14_ConBimamba

# Inference using HTSAT model
python inference.py --exp EXP_HTSAT --checkpoints_dir pretrained_ckpts/HTSAT
```

### Training

**Prerequisites:**
- Complete dataset setup (see Dataset section)
- Extract features first (see Feature Extraction section)


Train with specific experiment configurations:

```bash
# Train with HTSAT backbone (Hierarchical Token-Semantic Audio Transformer)
python run_experiment.py --exp EXP_HTSAT

# Train with CNN14-Conformer backbone (CNN14 + Conformer decoder)
python run_experiment.py --exp EXP_CNN14_Conformer

# Train with CNN14-BiMamba backbone (CNN14 + Bidirectional Mamba decoder)
python run_experiment.py --exp EXP_CNN14_BiMamba

# Train with CNN14-BiMamba-AC backbone (CNN14 + BiMamba + Asymmetric Convolution)
python run_experiment.py --exp EXP_CNN14_BiMambaAC

# Train with CNN14-ConBimamba backbone (CNN14 + Conformer + BiMamba hybrid)
python run_experiment.py --exp EXP_CNN14_ConBimamba

# Train with CNN14-ConBiMamba-AC backbone (CNN14 + Conformer + BiMamba + AC)
python run_experiment.py --exp EXP_CNN14_ConBiMambaAC
```




### Model Analysis

Calculate model parameters and FLOPs:

```bash
# Using ptflops (recommended)
python parameters_macs_calculation_ptflops.py
```



## Cite
[1] Wenmiao Gao and Yang Xiao. "Stereo sound event localization and detection based on PSELDnet pretraining and BiMamba sequence modeling."  [arXiv:2506.13455](https://arxiv.org/abs/2506.13455), 2025.

[2] Jinbo Hu, Yin Cao, Ming Wu, Fang Kang, Feiran Yang, Wenwu Wang, Mark D. Plumbley, Jun Yang, "PSELDNets: Pre-trained Neural Networks on Large-scale Synthetic Datasets for Sound Event Localization and Detection" [arXiv:2411.06399](https://arxiv.org/abs/2411.06399), 2024.

## References

1. **PSELDNets**: [https://github.com/Jinbo-Hu/PSELDNets](https://github.com/Jinbo-Hu/PSELDNets)
2. **HTS-AT**: [https://github.com/RetroCirce/HTS-Audio-Transformer](https://github.com/RetroCirce/HTS-Audio-Transformer)
3. **PANNs**: [https://github.com/qiuqiangkong/audioset_tagging_cnn](https://github.com/qiuqiangkong/audioset_tagging_cnn)
4. **DCASE 2025 Task 3**: [https://zenodo.org/records/15559774](https://zenodo.org/records/15559774)
