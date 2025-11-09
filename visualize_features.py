import os
import numpy as np
import cv2
import matplotlib.pyplot as plt
from pathlib import Path
import random

# --- Configuration ---
FEATURES_DIR = Path("data/features")
OUTPUT_DIR = Path("data/visualizations")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# --- Helper functions ---
def load_image(path):
    """Load image using OpenCV and convert to RGB."""
    img = cv2.imread(str(path))
    if img is None:
        raise FileNotFoundError(f"Image not found: {path}")
    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

def normalize_for_vis(array):
    """Normalize numpy array for visualization."""
    arr = array.astype(np.float32)
    arr = np.log(np.abs(arr) + 1e-3) if np.iscomplexobj(arr) else arr
    arr = (arr - arr.min()) / (arr.max() - arr.min() + 1e-8)
    return arr

def visualize_sample(label="real", subset="sample", frame_idx=0):
    """Visualize one RGB frame, residual, and FFT magnitude."""
    frame_dir = FEATURES_DIR / "frames" / label / subset
    fft_dir = FEATURES_DIR / "fft" / label / subset
    res_dir = FEATURES_DIR / "residuals" / label / subset

    frame_path = frame_dir / f"frame_{frame_idx:05d}.jpg"
    fft_path = fft_dir / f"frame_{frame_idx:05d}_fft.npy"
    res_path = res_dir / f"frame_{frame_idx:05d}_residual.npy"

    rgb = load_image(frame_path)
    fft = np.load(fft_path)
    res = np.load(res_path)

    fft_vis = normalize_for_vis(fft)
    res_vis = normalize_for_vis(res)

    plt.figure(figsize=(12, 4))
    plt.subplot(1, 3, 1)
    plt.imshow(rgb)
    plt.title("RGB Frame")
    plt.axis("off")

    plt.subplot(1, 3, 2)
    plt.imshow(res_vis, cmap="gray")
    plt.title("High-Pass Residual")
    plt.axis("off")

    plt.subplot(1, 3, 3)
    plt.imshow(fft_vis, cmap="magma")
    plt.title("FFT Magnitude Spectrum")
    plt.axis("off")

    plt.suptitle(f"{label.upper()} – {subset}", fontsize=14)
    out_path = OUTPUT_DIR / f"{label}_{subset}_frame{frame_idx}.png"
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()
    print(f"Saved visualization: {out_path}")

def main():
    print("=== Visualizing extracted features ===")
    label_options = ["real", "synthetic"]
    random.shuffle(label_options)

    for label in label_options:
        base_dir = FEATURES_DIR / "frames" / label
        if not base_dir.exists():
            print(f"No frames found for label '{label}'")
            continue

        # Pick a random subfolder (sample or dataset subset)
        subsets = [d.name for d in base_dir.iterdir() if d.is_dir()]
        if not subsets:
            continue

        subset = random.choice(subsets)
        visualize_sample(label=label, subset=subset, frame_idx=0)

    print("\nVisualization complete. Check data/visualizations/ for saved images.")

if __name__ == "__main__":
    main()
