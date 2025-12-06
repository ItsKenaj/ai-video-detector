# AI Video Detector

**Detecting AI-Generated Videos Using Spatial and Temporal Cues**  
*CS229: Machine Learning — Stanford University*  
Author: **Kenaj Washington** (kkenajj@stanford.edu)

---

## Overview

This project distinguishes **AI-generated videos** from **real videos** using frequency-domain forensics. We combine RGB frames with FFT magnitude spectra and high-frequency residual maps, feeding them into a modified ResNet-18. Our best model achieves **97.9% AUC** on video-level classification.

### Key Findings
- FFT + residual features provide **+13.8% AUC** improvement over RGB-only
- Optical flow features surprisingly **hurt** performance
- Frequency-domain artifacts generalize across different generators

---

## Project Structure

```
ai-video-detector/
├── data/
│   ├── real/                    # Real videos (Kinetics)
│   ├── synthetic/               # Synthetic videos (DeepAction)
│   ├── features/                # Extracted features (frames, FFT, residuals, flow)
│   └── visualizations/          # Generated figures
├── preprocessing/
│   ├── download_datasets.py     # Download real/synthetic videos
│   ├── run_preprocessing.py     # Run full preprocessing pipeline
│   ├── extract_frames.py        # Extract video frames
│   ├── compute_fft.py           # Compute FFT magnitude spectra
│   ├── compute_residuals.py     # Compute high-frequency residuals
│   └── compute_optical_flow.py  # Compute dense optical flow
├── scripts/
│   ├── build_video_manifest.py  # Build video metadata manifest
│   ├── create_splits.py         # Create train/val/test splits
│   ├── generate_report_figures.py
│   └── visualize_flow_comparison.py
├── training/
│   ├── dataset_loader.py        # PyTorch dataset for multimodal features
│   ├── train_baseline.py        # Train ResNet-18 models
│   ├── evaluate_videos.py       # Video-level evaluation
│   ├── logreg_baseline.py       # Logistic regression baseline
│   ├── generate_ablation_table.py
│   ├── cross_generator_experiment.py
│   └── single_generator_experiment.py
├── inference/
│   └── predict_video.py         # Inference on new videos
├── utils/
│   ├── config.py                # Configuration constants
│   └── plot_metrics.py          # Plotting utilities
├── results/                     # Checkpoints, plots, evaluation results
├── requirements.txt
└── README.md
```

---

## Quick Start

### 1. Setup Environment
```bash
git clone https://github.com/ItsKenaj/ai-video-detector.git
cd ai-video-detector
python -m venv venv
source venv/bin/activate  # or venv\Scripts\activate on Windows
pip install -r requirements.txt
```

### 2. Download Data
```bash
python preprocessing/download_datasets.py --n-real 100 --per-source 15
```

### 3. Preprocess Videos
```bash
python preprocessing/run_preprocessing.py
```

### 4. Build Splits
```bash
python scripts/build_video_manifest.py
python scripts/create_splits.py
```

### 5. Train Models
```bash
# Baseline (RGB + FFT + Residual)
python -m training.train_baseline

# With optical flow
python -m training.train_baseline --use-flow

# RGB only (ablation)
python -m training.train_baseline --rgb-only
```

### 6. Evaluate
```bash
python -m training.evaluate_videos --model resnet18.pt
python -m training.evaluate_videos --model resnet18_flow.pt --use-flow
python -m training.evaluate_videos --model resnet18_rgb.pt --rgb-only
```

### 7. Generate Results Table
```bash
python -m training.generate_ablation_table
```

---

## Results

| Model | Channels | AUC | Accuracy |
|-------|----------|-----|----------|
| Logistic Regression | 8 (features) | 0.733 | 0.645 |
| ResNet18 (RGB only) | 3 | 0.842 | 0.742 |
| ResNet18 (RGB+FFT+Res) | 5 | **0.979** | **0.903** |
| ResNet18 (+Flow) | 7 | 0.950 | 0.774 |

---

## Datasets

| Dataset | Type | Description |
|---------|------|-------------|
| [nateraw/kinetics](https://huggingface.co/datasets/nateraw/kinetics) | Real | Human action videos from Kinetics |
| [DeepAction](https://huggingface.co/datasets/ByteDance/DeepAction) | Synthetic | AI-generated videos from 7 generators |

---

## Requirements

- Python 3.8+
- PyTorch
- OpenCV
- NumPy
- scikit-learn
- matplotlib
- tqdm

See `requirements.txt` for full dependencies.

---

## Citation

If you use this code, please cite:
```
@misc{washington2024aivideo,
  author = {Washington, Kenaj},
  title = {Detecting AI-Generated Videos Using Spatial and Temporal Cues},
  year = {2025},
  institution = {Stanford University, CS229}
}
```
