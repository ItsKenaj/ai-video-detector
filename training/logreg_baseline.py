# training/logreg_baseline.py
"""
Logistic Regression Baseline for AI Video Detection

This script implements a simple, interpretable baseline using hand-crafted
features from FFT and residual maps. This ties the project to CS229 course
material and demonstrates the value of deep learning over linear models.

Features per frame:
- FFT mean, std, energy (sum of magnitudes)
- Residual mean, variance
- RGB histogram entropy (R, G, B channels)

Video-level: Average frame features → Logistic Regression → Real/Synthetic
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


def frame_stats(fft_path, res_path, rgb_path):
    """
    Extract simple interpretable statistics from one frame.
    
    Returns 8-dimensional feature vector:
    [fft_mean, fft_std, fft_energy, res_mean, res_var, r_entropy, g_entropy, b_entropy]
    """
    # Load FFT and residual
    fft = np.load(fft_path)
    res = np.load(res_path)
    
    # Load RGB and compute histogram entropy
    rgb = cv2.imread(str(rgb_path))
    if rgb is None:
        return None
    
    r_hist = np.histogram(rgb[..., 2], bins=32, range=(0, 255))[0]  # OpenCV uses BGR
    g_hist = np.histogram(rgb[..., 1], bins=32, range=(0, 255))[0]
    b_hist = np.histogram(rgb[..., 0], bins=32, range=(0, 255))[0]
    
    # Add small epsilon to avoid log(0)
    r_ent = entropy(r_hist + 1e-8)
    g_ent = entropy(g_hist + 1e-8)
    b_ent = entropy(b_hist + 1e-8)
    
    return np.array([
        fft.mean(),
        fft.std(),
        np.sum(np.abs(fft)),  # FFT energy
        res.mean(),
        res.var(),
        r_ent,
        g_ent,
        b_ent
    ], dtype=np.float32)


def build_video_features(video_dir):
    """
    Aggregate frame-level statistics into a single video feature vector.
    Uses mean pooling across all frames.
    """
    video_dir = Path(video_dir)
    
    # Construct paths for other modalities
    rel = video_dir.relative_to("data/features/frames")
    fft_dir = Path("data/features/fft") / rel
    res_dir = Path("data/features/residuals") / rel
    
    feats = []
    for frame in sorted(video_dir.glob("frame_*.jpg")):
        base = frame.stem.replace("frame_", "")
        fft_path = fft_dir / f"frame_{base}_fft.npy"
        res_path = res_dir / f"frame_{base}_residual.npy"
        
        if not fft_path.exists() or not res_path.exists():
            continue
        
        stats = frame_stats(str(fft_path), str(res_path), str(frame))
        if stats is not None:
            feats.append(stats)
    
    if len(feats) == 0:
        return None
    
    # Video-level aggregation: mean of frame features
    return np.mean(feats, axis=0)


def load_split(split):
    """Load video IDs for a given split."""
    with open(f"splits/{split}_videos.txt") as f:
        return [line.strip() for line in f]


def load_manifest():
    """Load video manifest and return video_id -> label mapping."""
    with open("splits/video_manifest.json") as f:
        data = json.load(f)
    return {d["video_id"]: d["label"] for d in data}


def main():
    print("\n" + "=" * 60)
    print("LOGISTIC REGRESSION BASELINE")
    print("Features: FFT stats + Residual stats + RGB entropy")
    print("=" * 60)
    
    manifest = load_manifest()
    
    # Build features for each split
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
    
    # Standardize features (important for logistic regression)
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_val_scaled = scaler.transform(X_val)
    X_test_scaled = scaler.transform(X_test)
    
    # Train logistic regression
    print("\nTraining Logistic Regression...")
    clf = LogisticRegression(max_iter=1000, random_state=42)
    clf.fit(X_train_scaled, y_train)
    
    # Feature importance (coefficient magnitudes)
    feature_names = ["FFT_mean", "FFT_std", "FFT_energy", "Res_mean", "Res_var", 
                     "R_entropy", "G_entropy", "B_entropy"]
    
    print("\n📊 FEATURE COEFFICIENTS (importance):")
    print("-" * 40)
    coef_importance = sorted(zip(feature_names, clf.coef_[0]), key=lambda x: abs(x[1]), reverse=True)
    for name, coef in coef_importance:
        print(f"  {name:<15} {coef:+.4f}")
    
    # Evaluation function
    def eval_model(name, X, y):
        probs = clf.predict_proba(X)[:, 1]
        preds = clf.predict(X)
        auc = roc_auc_score(y, probs)
        acc = accuracy_score(y, preds)
        cm = confusion_matrix(y, preds)
        return auc, acc, cm
    
    # Evaluate on all splits
    print("\n" + "=" * 60)
    print("RESULTS")
    print("=" * 60)
    
    results = {}
    for name, X, y in [("Train", X_train_scaled, y_train), 
                        ("Val", X_val_scaled, y_val), 
                        ("Test", X_test_scaled, y_test)]:
        auc, acc, cm = eval_model(name, X, y)
        results[name.lower()] = {"auc": auc, "accuracy": acc, "confusion_matrix": cm.tolist()}
        print(f"\n{name}:")
        print(f"  AUC:      {auc:.4f}")
        print(f"  Accuracy: {acc:.4f}")
        print(f"  Confusion Matrix:")
        print(f"    [[{cm[0,0]:3d} {cm[0,1]:3d}]")
        print(f"     [{cm[1,0]:3d} {cm[1,1]:3d}]]")
    
    # Save results
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
    
    print(f"\n✅ Results saved to {out_dir}/metrics.json")
    
    # Summary comparison hint
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

