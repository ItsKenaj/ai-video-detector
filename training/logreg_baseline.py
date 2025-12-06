#!/usr/bin/env python3
"""
=============================================================================
Logistic Regression Baseline for AI Video Detection
=============================================================================

This script implements a simple, interpretable baseline using hand-crafted
features from FFT and residual maps. This ties the project to CS229 course
material and demonstrates the value of deep learning over linear models.

Feature Engineering (8 features per video):
    FFT Features:
        - fft_mean: Average FFT magnitude (overall frequency content)
        - fft_std: Standard deviation (frequency variation)
        - fft_energy: Sum of magnitudes (total spectral energy)
    
    Residual Features:
        - res_mean: Average residual value (noise level)
        - res_var: Residual variance (noise consistency)
    
    RGB Entropy Features:
        - r_entropy: Red channel histogram entropy
        - g_entropy: Green channel histogram entropy
        - b_entropy: Blue channel histogram entropy

Pipeline:
    1. For each frame: compute 8 statistics
    2. For each video: average frame statistics (mean pooling)
    3. Standardize features (zero mean, unit variance)
    4. Train logistic regression classifier
    5. Evaluate on test split

Usage:
    python -m training.logreg_baseline

Output:
    - results/video_eval/logreg_baseline/metrics.json
=============================================================================
"""

import os
import json
import numpy as np
from pathlib import Path
from tqdm import tqdm
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score, accuracy_score, confusion_matrix
from scipy.stats import entropy
import cv2


# =============================================================================
# Feature Extraction Functions
# =============================================================================
def frame_stats(fft_path, res_path, rgb_path):
    """
    Extract hand-crafted statistics from a single frame.
    
    This function computes 8 interpretable features that capture:
        - Frequency domain characteristics (FFT statistics)
        - High-frequency noise patterns (residual statistics)
        - Color distribution (RGB histogram entropy)
    
    Args:
        fft_path: Path to FFT magnitude numpy file
        res_path: Path to residual map numpy file
        rgb_path: Path to RGB frame image
    
    Returns:
        8-dimensional numpy array of features, or None if frame can't be loaded
        
    Feature Vector:
        [fft_mean, fft_std, fft_energy, res_mean, res_var, r_ent, g_ent, b_ent]
    """
    # Load precomputed FFT magnitude spectrum
    fft = np.load(fft_path)
    
    # Load precomputed residual map (high-frequency noise)
    res = np.load(res_path)
    
    # Load RGB image for entropy calculation
    rgb = cv2.imread(str(rgb_path))
    if rgb is None:
        return None
    
    # Compute histogram entropy for each color channel
    # Entropy measures "randomness" of pixel value distribution
    # AI-generated images often have different entropy patterns
    # Note: OpenCV uses BGR format, not RGB
    r_hist = np.histogram(rgb[..., 2], bins=32, range=(0, 255))[0]
    g_hist = np.histogram(rgb[..., 1], bins=32, range=(0, 255))[0]
    b_hist = np.histogram(rgb[..., 0], bins=32, range=(0, 255))[0]
    
    # Add epsilon to avoid log(0) in entropy calculation
    r_ent = entropy(r_hist + 1e-8)
    g_ent = entropy(g_hist + 1e-8)
    b_ent = entropy(b_hist + 1e-8)
    
    # Construct feature vector
    return np.array([
        fft.mean(),           # FFT mean: overall frequency magnitude
        fft.std(),            # FFT std: frequency variation
        np.sum(np.abs(fft)),  # FFT energy: total spectral power
        res.mean(),           # Residual mean: average noise level
        res.var(),            # Residual variance: noise consistency
        r_ent,                # Red channel entropy
        g_ent,                # Green channel entropy
        b_ent                 # Blue channel entropy
    ], dtype=np.float32)


def build_video_features(video_dir):
    """
    Aggregate frame-level statistics into a single video feature vector.
    
    For each frame in the video, computes the 8 statistics, then averages
    them across all frames to produce a single 8-dimensional representation.
    
    Args:
        video_dir: Path to video's frame directory
    
    Returns:
        8-dimensional numpy array (mean of frame features), or None if no valid frames
    """
    video_dir = Path(video_dir)
    
    # Construct paths for feature modalities using relative path mapping
    rel = video_dir.relative_to("data/features/frames")
    fft_dir = Path("data/features/fft") / rel
    res_dir = Path("data/features/residuals") / rel
    
    # Collect features from all frames
    feats = []
    for frame in sorted(video_dir.glob("frame_*.jpg")):
        # Extract frame number and construct feature paths
        base = frame.stem.replace("frame_", "")
        fft_path = fft_dir / f"frame_{base}_fft.npy"
        res_path = res_dir / f"frame_{base}_residual.npy"
        
        # Skip frames with missing features
        if not fft_path.exists() or not res_path.exists():
            continue
        
        stats = frame_stats(str(fft_path), str(res_path), str(frame))
        if stats is not None:
            feats.append(stats)
    
    if len(feats) == 0:
        return None
    
    # Video-level aggregation: mean pooling across frames
    return np.mean(feats, axis=0)


# =============================================================================
# Data Loading Functions
# =============================================================================
def load_split(split):
    """
    Load list of video IDs for a given split.
    
    Args:
        split: One of "train", "val", or "test"
    
    Returns:
        List of video directory paths
    """
    with open(f"splits/{split}_videos.txt") as f:
        return [line.strip() for line in f]


def load_manifest():
    """
    Load video manifest and return video_id -> label mapping.
    
    Returns:
        Dictionary mapping video_id to label (0=real, 1=synthetic)
    """
    with open("splits/video_manifest.json") as f:
        data = json.load(f)
    return {d["video_id"]: d["label"] for d in data}


# =============================================================================
# Main Function
# =============================================================================
def main():
    """
    Train and evaluate logistic regression baseline.
    
    Steps:
        1. Extract features for all videos in train/val/test splits
        2. Standardize features using training set statistics
        3. Train logistic regression classifier
        4. Evaluate on all splits and print results
        5. Save metrics to JSON file
    """
    print("\n" + "=" * 60)
    print("LOGISTIC REGRESSION BASELINE")
    print("Features: FFT stats + Residual stats + RGB entropy")
    print("=" * 60)
    
    manifest = load_manifest()
    
    # =========================================================================
    # Feature Extraction for Each Split
    # =========================================================================
    splits_data = {}
    for split in ["train", "val", "test"]:
        print(f"\nBuilding features for {split} split...")
        X, y = [], []
        videos = load_split(split)
        
        for video in tqdm(videos, desc=split):
            fv = build_video_features(video)
            if fv is not None:
                X.append(fv)
                y.append(manifest[video])
        
        splits_data[split] = (np.array(X), np.array(y))
        print(f"  {split}: {len(X)} videos ({sum(y)} synthetic, {len(y)-sum(y)} real)")
    
    X_train, y_train = splits_data["train"]
    X_val, y_val = splits_data["val"]
    X_test, y_test = splits_data["test"]
    
    # =========================================================================
    # Feature Standardization
    # =========================================================================
    # Standardization is important for logistic regression convergence
    # Fit scaler on training data only to avoid data leakage
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_val_scaled = scaler.transform(X_val)
    X_test_scaled = scaler.transform(X_test)
    
    # =========================================================================
    # Train Logistic Regression
    # =========================================================================
    print("\nTraining Logistic Regression...")
    clf = LogisticRegression(max_iter=1000, random_state=42)
    clf.fit(X_train_scaled, y_train)
    
    # =========================================================================
    # Analyze Feature Importance
    # =========================================================================
    feature_names = [
        "FFT_mean", "FFT_std", "FFT_energy", 
        "Res_mean", "Res_var", 
        "R_entropy", "G_entropy", "B_entropy"
    ]
    
    print("\nFEATURE COEFFICIENTS (importance):")
    print("-" * 40)
    
    # Sort features by absolute coefficient value (importance)
    coef_importance = sorted(
        zip(feature_names, clf.coef_[0]), 
        key=lambda x: abs(x[1]), 
        reverse=True
    )
    for name, coef in coef_importance:
        # Positive coefficient = feature increases P(synthetic)
        # Negative coefficient = feature decreases P(synthetic)
        print(f"  {name:<15} {coef:+.4f}")
    
    # =========================================================================
    # Evaluation
    # =========================================================================
    def eval_model(name, X, y):
        """Evaluate model on a dataset and return metrics."""
        probs = clf.predict_proba(X)[:, 1]  # Probability of synthetic
        preds = clf.predict(X)               # Binary predictions
        auc = roc_auc_score(y, probs)
        acc = accuracy_score(y, preds)
        cm = confusion_matrix(y, preds)
        return auc, acc, cm
    
    print("\n" + "=" * 60)
    print("RESULTS")
    print("=" * 60)
    
    results = {}
    for name, X, y in [("Train", X_train_scaled, y_train), 
                        ("Val", X_val_scaled, y_val), 
                        ("Test", X_test_scaled, y_test)]:
        auc, acc, cm = eval_model(name, X, y)
        results[name.lower()] = {
            "auc": auc, 
            "accuracy": acc, 
            "confusion_matrix": cm.tolist()
        }
        print(f"\n{name}:")
        print(f"  AUC:      {auc:.4f}")
        print(f"  Accuracy: {acc:.4f}")
        print(f"  Confusion Matrix:")
        print(f"              Predicted")
        print(f"              Real  Synth")
        print(f"    Actual Real   {cm[0,0]:3d}   {cm[0,1]:3d}")
        print(f"           Synth  {cm[1,0]:3d}   {cm[1,1]:3d}")
    
    # =========================================================================
    # Save Results
    # =========================================================================
    out_dir = Path("results/video_eval/logreg_baseline")
    out_dir.mkdir(parents=True, exist_ok=True)
    
    with open(out_dir / "metrics.json", "w") as f:
        json.dump({
            "model": "LogisticRegression",
            "features": feature_names,
            "test_auc": results["test"]["auc"],
            "test_accuracy": results["test"]["accuracy"],
            "test_confusion_matrix": results["test"]["confusion_matrix"],
            "train_auc": results["train"]["auc"],
            "val_auc": results["val"]["auc"],
            "coefficients": dict(zip(feature_names, clf.coef_[0].tolist()))
        }, f, indent=2)
    
    print(f"\nResults saved to {out_dir}/metrics.json")
    
    # =========================================================================
    # Comparison Summary
    # =========================================================================
    print("\n" + "=" * 60)
    print("COMPARISON WITH DEEP MODELS")
    print("=" * 60)
    print(f"""
Expected comparison (update with your actual results):

| Model                        | Channels | Test AUC | Test Acc |
|------------------------------|----------|----------|----------|
| Logistic Regression          |    8*    | {results['test']['auc']:.4f}   | {results['test']['accuracy']:.4f}   |
| ResNet18 (RGB-only)          |    3     |   ???    |   ???    |
| ResNet18 (RGB+FFT+Res)       |    5     |  0.9792  |  0.9032  |
| ResNet18 (RGB+FFT+Res+Flow)  |    7     |  0.9500  |  0.7742  |

* LogReg uses 8 hand-crafted features (not raw pixels)

This demonstrates the value of deep learning for this task!
""")


if __name__ == "__main__":
    main()
