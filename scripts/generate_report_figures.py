#!/usr/bin/env python3
"""
Generate all visualizations needed for the CS229 report.
Run this script on the VM after preprocessing is complete.

Outputs:
    - data/visualizations/fft_comparison.png
    - data/visualizations/score_hist.png (copied from results)
"""

import cv2
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import shutil


def generate_fft_comparison():
    """Create side-by-side FFT comparison of real vs synthetic frames."""
    print("Generating FFT comparison...")
    
    # Find real samples
    real_frames = list(Path('data/features/frames/real').rglob('frame_00001.jpg'))
    real_ffts = list(Path('data/features/fft/real').rglob('frame_00001_fft.npy'))
    
    # Find synthetic samples
    syn_frames = list(Path('data/features/frames/synthetic').rglob('frame_00001.jpg'))
    syn_ffts = list(Path('data/features/fft/synthetic').rglob('frame_00001_fft.npy'))
    
    if not real_frames or not syn_frames:
        print("  Error: Could not find frame samples")
        return
    
    if not real_ffts or not syn_ffts:
        print("  Error: Could not find FFT samples")
        return
    
    # Create figure
    fig, axes = plt.subplots(2, 2, figsize=(10, 8))
    
    # Real frame and FFT
    real_img = cv2.cvtColor(cv2.imread(str(real_frames[0])), cv2.COLOR_BGR2RGB)
    axes[0, 0].imshow(real_img)
    axes[0, 0].set_title('Real Frame', fontsize=14, fontweight='bold')
    axes[0, 0].axis('off')
    
    real_fft_data = np.load(real_ffts[0])
    axes[0, 1].imshow(real_fft_data, cmap='viridis')
    axes[0, 1].set_title('Real FFT Spectrum', fontsize=14, fontweight='bold')
    axes[0, 1].axis('off')
    
    # Synthetic frame and FFT
    syn_img = cv2.cvtColor(cv2.imread(str(syn_frames[0])), cv2.COLOR_BGR2RGB)
    axes[1, 0].imshow(syn_img)
    axes[1, 0].set_title('Synthetic Frame', fontsize=14, fontweight='bold')
    axes[1, 0].axis('off')
    
    syn_fft_data = np.load(syn_ffts[0])
    axes[1, 1].imshow(syn_fft_data, cmap='viridis')
    axes[1, 1].set_title('Synthetic FFT Spectrum', fontsize=14, fontweight='bold')
    axes[1, 1].axis('off')
    
    plt.tight_layout()
    
    out_path = Path('data/visualizations/fft_comparison.png')
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"  Saved: {out_path}")


def copy_score_histogram():
    """Copy score histogram from evaluation results."""
    print("Copying score histogram...")
    
    src = Path('results/video_eval/resnet18/score_hist.png')
    dst = Path('data/visualizations/score_hist.png')
    
    if src.exists():
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy(src, dst)
        print(f"  Saved: {dst}")
    else:
        print(f"  Warning: {src} not found. Run evaluation first.")


def main():
    print("\n" + "=" * 50)
    print("GENERATING REPORT FIGURES")
    print("=" * 50 + "\n")
    
    generate_fft_comparison()
    copy_score_histogram()
    
    print("\n" + "=" * 50)
    print("Done! Figures saved to data/visualizations/")
    print("=" * 50)
    
    # List generated files
    viz_dir = Path('data/visualizations')
    if viz_dir.exists():
        print("\nGenerated files:")
        for f in sorted(viz_dir.glob('*.png')):
            print(f"  - {f}")


if __name__ == "__main__":
    main()

