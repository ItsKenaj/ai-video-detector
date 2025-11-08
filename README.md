# AI Video Detector
Detecting AI-Generated Videos Using Spatio-Temporal Forensics  
*CS229: Machine Learning — Stanford University*  
Author: **Kenaj Washington**

---

## Overview
This project aims to distinguish **AI-generated videos** from **real human-captured videos** using spatial–temporal forensics.  
It combines deep learning, frequency-domain analysis, and motion features to capture subtle generative artifacts that modern diffusion-based video models produce.

---

## Objectives
- Build a **baseline model** (ResNet-18 / ViT-tiny) for frame-level detection.  
- Integrate **temporal cues** (optical flow, flicker, frequency residuals).  
- Evaluate **robustness** under compression, noise, and re-encoding.  
- Visualize **false positives/negatives** to interpret model reasoning.

---

## Datasets

| Dataset | Type | Description | Link |
|----------|------|-------------|------|
| **Kinetics-400 Subset** | Real | Natural videos of human activity. | [GitHub](https://github.com/cvdfoundation/kinetics-dataset) |
| **DeepAction_v1** | Synthetic | 2,600 AI-generated videos from multiple diffusion models. | [Hugging Face](https://huggingface.co/datasets/faridlab/deepaction_v1) |
| **SeeTrails AIGVDet v2.0.0** | Synthetic | Diffusion-based generated video dataset for detection tasks. | [Hugging Face](https://huggingface.co/kalpitbcontrails/seetrails_aigvdet_v2.0.0) |
| **GenVideo-100K** | Synthetic | 100,000 text-to-video diffusion clips for generation/detection research. | [ModelScope](https://modelscope.cn/datasets/cccnju/GenVideo-100K) |

---

## Repository Structure

```
ai-video-detector/
├── data/                 # real & synthetic datasets
│   ├── real/
│   └── synthetic/
├── preprocessing/        # frame extraction, FFT, residuals, dataset downloaders
├── models/               # model definitions (ResNet, TimeSformer, etc.)
├── utils/                # helper functions (configs, metrics, plots)
├── results/              # training logs, ROC curves, saved checkpoints
└── main.py               # pipeline entry point
```

## Setup Instructions

### 1. Clone and Initialize
```bash
git clone https://github.com/ItsKenaj/ai-video-detector.git
cd ai-video-detector
python -m venv venv
venv\Scripts\activate   # (Windows)
# or source venv/bin/activate   # (macOS/Linux)
pip install -r requirements.txt
```

---

### 2. Download Datasets
```bash
python preprocessing/download_datasets.py
```

---

### 3. Preprocess Data
Extract frames, compute residuals, and FFT features:
```bash
python preprocessing/extract_frames.py
python preprocessing/compute_residuals.py
python preprocessing/compute_fft.py
```

---

### 4. Train and Evaluate
Train your baseline model (ResNet-18) and evaluate performance:
```bash
python main.py --train
```

Trained weights, logs, and figures will be saved in:
```
results/logs/
results/figures/
```
