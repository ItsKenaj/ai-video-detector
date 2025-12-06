#!/usr/bin/env python3
"""
Generate optical flow comparison visualizations for the report.
Creates a grid showing frames and flow for real vs different AI generators.
"""

import cv2
import numpy as np
from pathlib import Path
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend
import matplotlib.pyplot as plt
from PIL import Image

def load_flow_as_image(flow_path):
    """Convert optical flow to HSV color visualization."""
    flow = np.load(flow_path)
    
    # Compute magnitude and angle
    mag, ang = cv2.cartToPolar(flow[..., 0], flow[..., 1])
    
    # Create HSV image
    hsv = np.zeros((flow.shape[0], flow.shape[1], 3), dtype=np.uint8)
    hsv[..., 0] = ang * 180 / np.pi / 2  # Hue = angle
    hsv[..., 1] = 255  # Saturation = max
    hsv[..., 2] = cv2.normalize(mag, None, 0, 255, cv2.NORM_MINMAX)  # Value = magnitude
    
    # Convert to RGB
    rgb = cv2.cvtColor(hsv, cv2.COLOR_HSV2RGB)
    return rgb


def find_sample_videos():
    """Find one sample video from each generator and real."""
    samples = {}
    
    # Real video
    real_dirs = list(Path("data/features/frames/real").rglob("frame_00001.jpg"))
    if real_dirs:
        samples["Real"] = real_dirs[0].parent
    
    # Synthetic videos by generator
    synthetic_root = Path("data/features/frames/synthetic/deepaction_v1")
    if synthetic_root.exists():
        for gen_dir in synthetic_root.iterdir():
            if gen_dir.is_dir():
                # Find first video with frames
                for video_dir in gen_dir.rglob("frame_00001.jpg"):
                    samples[gen_dir.name] = video_dir.parent
                    break
    
    return samples


def main():
    samples = find_sample_videos()
    
    if not samples:
        print("No samples found. Make sure preprocessing is complete.")
        return
    
    print(f"Found samples for: {list(samples.keys())}")
    
    # Select up to 4 generators for comparison
    generators_to_show = ["Real"]
    for gen in ["Veo", "RunwayML", "StableDiffusion", "CogVideoX5B", "VideoPoet"]:
        if gen in samples and len(generators_to_show) < 4:
            generators_to_show.append(gen)
    
    # Fill remaining slots
    for gen in samples.keys():
        if gen not in generators_to_show and len(generators_to_show) < 4:
            generators_to_show.append(gen)
    
    print(f"Showing: {generators_to_show}")
    
    # Create figure
    fig, axes = plt.subplots(2, len(generators_to_show), figsize=(4*len(generators_to_show), 6))
    
    for i, gen in enumerate(generators_to_show):
        video_dir = samples[gen]
        
        # Get frame path
        frame_path = video_dir / "frame_00001.jpg"
        
        # Get flow path
        rel = video_dir.relative_to("data/features/frames")
        flow_dir = Path("data/features/flow") / rel
        flow_path = flow_dir / "frame_00001_flow.npy"
        
        # Load and display frame
        if frame_path.exists():
            frame = cv2.imread(str(frame_path))
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            axes[0, i].imshow(frame)
        axes[0, i].set_title(gen, fontsize=12, fontweight='bold')
        axes[0, i].axis('off')
        
        # Load and display flow
        if flow_path.exists():
            flow_vis = load_flow_as_image(flow_path)
            axes[1, i].imshow(flow_vis)
        else:
            axes[1, i].text(0.5, 0.5, 'No flow', ha='center', va='center', transform=axes[1, i].transAxes)
        axes[1, i].axis('off')
    
    # Add row labels
    axes[0, 0].set_ylabel('Frame', fontsize=12)
    axes[1, 0].set_ylabel('Optical Flow', fontsize=12)
    
    plt.tight_layout()
    
    # Save
    out_path = Path("data/visualizations/flow_comparison.png")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"Saved: {out_path}")


if __name__ == "__main__":
    main()

