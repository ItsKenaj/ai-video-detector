import os
import numpy as np
import cv2
import matplotlib
matplotlib.use("Agg")  # ensures it runs headless (no display required)
import matplotlib.pyplot as plt
from pathlib import Path
import random

FEATURES_DIR = Path("data/features")
OUTPUT_DIR = Path("data/visualizations")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

def load_image(path):
    img = cv2.imread(str(path))
    if img is None:
        raise FileNotFoundError(f"Image not found: {path}")
    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

def normalize_for_vis(array):
    arr = array.astype(np.float32)
    arr = np.log(np.abs(arr) + 1e-3) if np.iscomplexobj(arr) else arr
    arr = (arr - arr.min()) / (arr.max() - arr.min() + 1e-8)
    return arr

def visualize_sample(frame_dir, fft_dir, res_dir, label, subset, frame_idx=0):
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
    labels = ["real", "synthetic"]

    for label in labels:
        base_dir = FEATURES_DIR / "frames" / label
        if not base_dir.exists():
            print(f"No frames found for '{label}'")
            continue

        # Find all nested subdirectories containing frames
        all_subsets = [p for p in base_dir.rglob("frame_00000.jpg")]
        if not all_subsets:
            print(f"No frame_00000.jpg found under {base_dir}")
            continue

        sample_frame = random.choice(all_subsets)
        subset_dir = sample_frame.parent
        rel_subset = subset_dir.relative_to(base_dir)
        subset_name = str(rel_subset).replace("/", "_")

        # Match FFT and residual directories by the same relative path
        fft_dir = FEATURES_DIR / "fft" / label / rel_subset
        res_dir = FEATURES_DIR / "residuals" / label / rel_subset

        visualize_sample(subset_dir, fft_dir, res_dir, label, subset_name, frame_idx=0)

    print("\nVisualization complete. Check data/visualizations/ for saved images.")

if __name__ == "__main__":
    main()
