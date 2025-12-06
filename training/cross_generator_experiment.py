#!/usr/bin/env python3
"""
=============================================================================
Cross-Generator Generalization Experiment
=============================================================================

This script implements a leave-one-generator-out experiment to evaluate
whether a model trained on some AI video generators can detect videos
from unseen generators.

Experiment Design:
    For each generator G in {Veo, RunwayML, CogVideoX5B, ...}:
        1. TRAIN on: all real videos + all synthetic videos EXCEPT from G
        2. TEST on: synthetic videos from G + subset of real videos
        3. RECORD: AUC and accuracy for generator G

Key Research Questions:
    - Do FFT+Residual features capture generator-agnostic artifacts?
    - Which generators are hardest/easiest to detect?
    - Does the model overfit to specific generator patterns?

Output:
    - Per-generator AUC and accuracy
    - Average generalization performance
    - LaTeX table for paper
    - JSON results file

Usage:
    python -m training.cross_generator_experiment
=============================================================================
"""

import json
import os
import random
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
from sklearn.metrics import roc_auc_score, accuracy_score
import torchvision.models as models
from PIL import Image
import cv2

# =============================================================================
# Configuration
# =============================================================================
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
EPOCHS = 3      # Fewer epochs for faster experiments (7 models to train)
BATCH_SIZE = 16
LR = 1e-4       # Learning rate for Adam optimizer


# =============================================================================
# Dataset Class
# =============================================================================
class CrossGenDataset(Dataset):
    """
    Custom PyTorch Dataset for cross-generator experiments.
    
    Unlike VideoSplitFrameDataset, this class accepts an arbitrary list of
    video directories rather than reading from split files. This allows
    flexible train/test splits for leave-one-out experiments.
    
    Args:
        video_list: List of video directory paths (e.g., 'data/features/frames/synthetic/...')
        manifest: Dictionary mapping video_id -> {label, generator, ...}
        use_flow: Whether to include optical flow (default: False)
    
    Returns per sample:
        x: Tensor of shape [5, 224, 224] containing [RGB(3) + FFT(1) + Residual(1)]
        y: Label (0=real, 1=synthetic)
    """
    
    def __init__(self, video_list, manifest, use_flow=False):
        self.samples = []
        self.use_flow = use_flow
        
        # Iterate through each video directory and collect valid frame samples
        for video_dir in video_list:
            label = manifest[video_dir]["label"]
            frame_root = Path(video_dir)
            
            # Construct paths for other modalities using relative path mapping
            # Example: data/features/frames/synthetic/... -> data/features/fft/synthetic/...
            rel = frame_root.relative_to("data/features/frames")
            fft_root = Path("data/features/fft") / rel
            res_root = Path("data/features/residuals") / rel
            flow_root = Path("data/features/flow") / rel
            
            # Find all frame files and check if corresponding features exist
            for frame_path in sorted(frame_root.glob("frame_*.jpg")):
                base = frame_path.stem.replace("frame_", "")  # Extract frame number
                fft_path = fft_root / f"frame_{base}_fft.npy"
                res_path = res_root / f"frame_{base}_residual.npy"
                flow_path = flow_root / f"frame_{base}_flow.npy"
                
                # Only include samples where all required features exist
                if fft_path.exists() and res_path.exists():
                    if not self.use_flow or flow_path.exists():
                        self.samples.append((frame_path, fft_path, res_path, flow_path, label))
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        """
        Load and preprocess a single frame with all modalities.
        
        Processing steps:
            1. Load RGB image, normalize to [0, 1]
            2. Load FFT magnitude spectrum (numpy array)
            3. Load residual map (numpy array)
            4. Resize all to 224x224 for ResNet
            5. Concatenate into 5-channel tensor
            6. Convert to PyTorch tensor (channels-first format)
        """
        frame_path, fft_path, res_path, flow_path, label = self.samples[idx]
        
        # Load and preprocess RGB image
        rgb = np.array(Image.open(frame_path).convert("RGB")).astype(np.float32) / 255.0
        rgb = cv2.resize(rgb, (224, 224), interpolation=cv2.INTER_AREA)
        
        # Load FFT and residual features (stored as numpy arrays)
        fft = np.load(fft_path).astype(np.float32)
        res = np.load(res_path).astype(np.float32)
        
        # Resize features to match RGB dimensions
        fft = cv2.resize(fft, (224, 224), interpolation=cv2.INTER_AREA)
        res = cv2.resize(res, (224, 224), interpolation=cv2.INTER_AREA)
        
        # Concatenate modalities: RGB(3) + FFT(1) + Residual(1) = 5 channels
        # FFT and residual are single-channel, so we add a dimension
        x = np.concatenate([rgb, fft[..., None], res[..., None]], axis=2)
        
        # Convert to PyTorch format: [H, W, C] -> [C, H, W]
        x = np.transpose(x, (2, 0, 1))
        x = torch.from_numpy(x).float()
        y = torch.tensor(label, dtype=torch.float32)
        
        return x, y


# =============================================================================
# Data Loading Utilities
# =============================================================================
def load_manifest():
    """
    Load the video manifest containing metadata for all videos.
    
    Returns:
        Dictionary mapping video_id -> {video_id, label, generator, num_frames}
    """
    with open("splits/video_manifest.json") as f:
        data = json.load(f)
    return {d["video_id"]: d for d in data}


def get_generators(manifest):
    """
    Extract list of unique AI generators from the manifest.
    
    Args:
        manifest: Video manifest dictionary
    
    Returns:
        Sorted list of generator names (e.g., ['CogVideoX5B', 'RunwayML', 'Veo', ...])
    """
    generators = set()
    for vid, info in manifest.items():
        if info["label"] == 1:  # Only synthetic videos have generators
            generators.add(info["generator"])
    return sorted(list(generators))


def get_videos_by_generator(manifest):
    """
    Group video IDs by their generator (or 'real' for real videos).
    
    Args:
        manifest: Video manifest dictionary
    
    Returns:
        Dictionary mapping generator_name -> list of video_ids
        Special key 'real' contains all real videos
    """
    by_gen = {"real": []}
    for vid, info in manifest.items():
        if info["label"] == 0:
            by_gen["real"].append(vid)
        else:
            gen = info["generator"]
            if gen not in by_gen:
                by_gen[gen] = []
            by_gen[gen].append(vid)
    return by_gen


# =============================================================================
# Model Utilities
# =============================================================================
def get_model(in_channels=5):
    """
    Create a ResNet18 model modified for multi-channel forensic input.
    
    Modifications from standard ResNet18:
        1. First conv layer accepts `in_channels` instead of 3
        2. Final FC layer outputs 1 (binary classification logit)
        3. Uses ImageNet pretrained weights for transfer learning
    
    Args:
        in_channels: Number of input channels (5 for RGB+FFT+Res)
    
    Returns:
        Modified ResNet18 model
    """
    model = models.resnet18(weights="IMAGENET1K_V1")
    
    # Modify first conv to accept multi-channel input
    model.conv1 = nn.Conv2d(in_channels, 64, kernel_size=7, stride=2, padding=3, bias=False)
    
    # Modify final layer for binary classification
    model.fc = nn.Linear(model.fc.in_features, 1)
    
    return model


def train_model(train_videos, manifest, epochs=EPOCHS):
    """
    Train a ResNet18 model on the given video list.
    
    Args:
        train_videos: List of video directory paths for training
        manifest: Video manifest for label lookup
        epochs: Number of training epochs
    
    Returns:
        Trained PyTorch model
    """
    # Create dataset and dataloader
    train_ds = CrossGenDataset(train_videos, manifest)
    train_dl = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, num_workers=2)
    
    # Initialize model, loss function, and optimizer
    model = get_model(in_channels=5).to(DEVICE)
    criterion = nn.BCEWithLogitsLoss()  # Binary cross-entropy with logits
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)
    
    # Training loop
    model.train()
    for epoch in range(epochs):
        epoch_loss = 0.0
        for x, y in train_dl:
            # Move data to GPU if available
            x, y = x.to(DEVICE), y.to(DEVICE).unsqueeze(1)
            
            # Forward pass
            optimizer.zero_grad()
            logits = model(x)
            loss = criterion(logits, y)
            
            # Backward pass and optimization
            loss.backward()
            optimizer.step()
            
            epoch_loss += loss.item()
    
    return model


def evaluate_model(model, test_videos, manifest):
    """
    Evaluate model on test videos using video-level aggregation.
    
    Evaluation process:
        1. Run inference on all frames from test videos
        2. Group predictions by video (parent directory)
        3. Compute mean probability per video (aggregation)
        4. Calculate video-level AUC and accuracy
    
    Args:
        model: Trained PyTorch model
        test_videos: List of video directory paths for testing
        manifest: Video manifest for label lookup
    
    Returns:
        Tuple of (auc, accuracy, num_videos)
        Note: auc may be None if only one class is present in test set
    """
    test_ds = CrossGenDataset(test_videos, manifest)
    
    # Handle empty test set
    if len(test_ds) == 0:
        return None, None, 0
    
    # Use shuffle=False and num_workers=0 for deterministic sample ordering
    test_dl = DataLoader(test_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)
    
    model.eval()
    video_scores = {}   # video_dir -> list of frame probabilities
    video_labels = {}   # video_dir -> label
    
    sample_idx = 0  # Track current sample for correct mapping
    with torch.no_grad():
        for x, y in test_dl:
            x = x.to(DEVICE)
            logits = model(x)
            probs = torch.sigmoid(logits).cpu().numpy().ravel()
            
            # Map each prediction to its video using tracked index
            for prob in probs:
                frame_path, _, _, _, label = test_ds.samples[sample_idx]
                video_dir = str(Path(frame_path).parent)
                
                if video_dir not in video_scores:
                    video_scores[video_dir] = []
                    video_labels[video_dir] = label
                video_scores[video_dir].append(prob)
                sample_idx += 1
    
    # Aggregate frame predictions to video level using mean
    final_scores = []
    final_labels = []
    for vid, scores in video_scores.items():
        final_scores.append(float(np.mean(scores)))
        final_labels.append(video_labels[vid])
    
    # Handle case where only one class is present (can't compute AUC)
    if len(set(final_labels)) < 2:
        preds = (np.array(final_scores) > 0.5).astype(int)
        acc = accuracy_score(final_labels, preds)
        return None, acc, len(final_labels)
    
    # Compute metrics
    auc = roc_auc_score(final_labels, final_scores)
    preds = (np.array(final_scores) > 0.5).astype(int)
    acc = accuracy_score(final_labels, preds)
    
    return auc, acc, len(final_labels)


# =============================================================================
# Main Experiment
# =============================================================================
def main():
    """
    Run the complete cross-generator generalization experiment.
    
    Experiment loop:
        For each generator G:
            1. Create train set: all real + synthetic (except G)
            2. Create test set: G synthetic + subset of real
            3. Train model from scratch
            4. Evaluate on test set
            5. Record results
    """
    print("\n" + "=" * 70)
    print("CROSS-GENERATOR GENERALIZATION EXPERIMENT")
    print("Leave-One-Generator-Out Training")
    print("=" * 70)
    
    # Set random seeds for reproducibility
    random.seed(42)
    torch.manual_seed(42)
    
    # Load data
    manifest = load_manifest()
    videos_by_gen = get_videos_by_generator(manifest)
    generators = [g for g in videos_by_gen.keys() if g != "real"]
    
    # Print dataset statistics
    print(f"\nFound {len(generators)} generators:")
    for gen in generators:
        print(f"  - {gen}: {len(videos_by_gen[gen])} videos")
    print(f"  - real: {len(videos_by_gen['real'])} videos")
    
    results = []
    
    # Main experiment loop: iterate through each generator as held-out
    for held_out_gen in generators:
        print(f"\n{'='*70}")
        print(f"HELD OUT: {held_out_gen}")
        print(f"{'='*70}")
        
        # Build training set: all real + all synthetic EXCEPT held_out_gen
        train_videos = videos_by_gen["real"].copy()
        for gen in generators:
            if gen != held_out_gen:
                train_videos.extend(videos_by_gen[gen])
        
        # Build test set: held_out_gen + subset of real (need both classes for AUC)
        # Use 1/3 of real videos for testing to ensure class balance
        real_test = random.sample(videos_by_gen["real"], len(videos_by_gen["real"]) // 3)
        test_videos = videos_by_gen[held_out_gen] + real_test
        
        # Remove test real videos from training to avoid leakage
        train_videos = [v for v in train_videos if v not in real_test]
        
        print(f"Train: {len(train_videos)} videos")
        print(f"Test: {len(test_videos)} videos ({len(videos_by_gen[held_out_gen])} {held_out_gen} + {len(real_test)} real)")
        
        # Train model
        print(f"Training for {EPOCHS} epochs...")
        model = train_model(train_videos, manifest, epochs=EPOCHS)
        
        # Evaluate
        print("Evaluating on held-out generator...")
        auc, acc, n_videos = evaluate_model(model, test_videos, manifest)
        
        # Record results
        if auc is not None:
            print(f"Results: AUC={auc:.4f}, Acc={acc:.4f}")
            results.append({
                "generator": held_out_gen,
                "auc": auc,
                "accuracy": acc,
                "n_test_videos": n_videos
            })
        else:
            print(f"Results: AUC=N/A (single class), Acc={acc:.4f}")
            results.append({
                "generator": held_out_gen,
                "auc": None,
                "accuracy": acc,
                "n_test_videos": n_videos
            })
    
    # ==========================================================================
    # Print Summary Results
    # ==========================================================================
    print("\n" + "=" * 70)
    print("CROSS-GENERATOR GENERALIZATION RESULTS")
    print("=" * 70)
    print(f"\n{'Held-Out Generator':<25} {'AUC':<10} {'Accuracy':<10} {'N Videos':<10}")
    print("-" * 55)
    
    valid_aucs = []
    for r in results:
        auc_str = f"{r['auc']:.4f}" if r['auc'] is not None else "N/A"
        print(f"{r['generator']:<25} {auc_str:<10} {r['accuracy']:.4f}     {r['n_test_videos']:<10}")
        if r['auc'] is not None:
            valid_aucs.append(r['auc'])
    
    print("-" * 55)
    if valid_aucs:
        avg_auc = np.mean(valid_aucs)
        std_auc = np.std(valid_aucs)
        print(f"{'AVERAGE':<25} {avg_auc:.4f}     (±{std_auc:.4f})")
    
    # ==========================================================================
    # Analysis and Interpretation
    # ==========================================================================
    print("\nANALYSIS:")
    if valid_aucs:
        best = max(results, key=lambda x: x['auc'] if x['auc'] else 0)
        worst = min([r for r in results if r['auc']], key=lambda x: x['auc'])
        print(f"  Easiest to detect: {best['generator']} (AUC={best['auc']:.4f})")
        print(f"  Hardest to detect: {worst['generator']} (AUC={worst['auc']:.4f})")
        print(f"  Average generalization: {avg_auc:.4f} +/- {std_auc:.4f}")
        
        # Interpret generalization quality
        if avg_auc > 0.85:
            print("\n  [GOOD] Model generalizes well across generators")
            print("         FFT+Residual features capture generator-agnostic artifacts")
        elif avg_auc > 0.70:
            print("\n  [MODERATE] Moderate generalization across generators")
            print("             Some generator-specific patterns may exist")
        else:
            print("\n  [POOR] Poor generalization across generators")
            print("         Model may be overfitting to specific generator artifacts")
    
    # ==========================================================================
    # Save Results
    # ==========================================================================
    out_dir = Path("results/cross_generator")
    out_dir.mkdir(parents=True, exist_ok=True)
    
    with open(out_dir / "results.json", "w") as f:
        json.dump({
            "experiment": "leave-one-generator-out",
            "epochs": EPOCHS,
            "results": results,
            "average_auc": float(np.mean(valid_aucs)) if valid_aucs else None,
            "std_auc": float(np.std(valid_aucs)) if valid_aucs else None
        }, f, indent=2)
    
    print(f"\nResults saved to {out_dir}/results.json")
    
    # ==========================================================================
    # LaTeX Table Output
    # ==========================================================================
    print("\nLATEX TABLE (copy to paper):")
    print("-" * 55)
    print(r"\begin{table}[h]")
    print(r"\centering")
    print(r"\begin{tabular}{lcc}")
    print(r"\toprule")
    print(r"Held-Out Generator & AUC & Accuracy \\")
    print(r"\midrule")
    for r in results:
        auc_str = f"{r['auc']:.4f}" if r['auc'] else "N/A"
        print(f"{r['generator']} & {auc_str} & {r['accuracy']:.4f} \\\\")
    print(r"\midrule")
    if valid_aucs:
        print(f"Average & {np.mean(valid_aucs):.4f} & -- \\\\")
    print(r"\bottomrule")
    print(r"\end{tabular}")
    print(r"\caption{Cross-generator generalization results. Model trained on N-1 generators, tested on held-out generator.}")
    print(r"\label{tab:cross-gen}")
    print(r"\end{table}")
    print("-" * 55)


if __name__ == "__main__":
    main()
