import cv2
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path


def flow_to_rgb(flow):
    """Convert optical flow (dx, dy) to a color-coded RGB map."""
    h, w = flow.shape[:2]
    hsv = np.zeros((h, w, 3), dtype=np.float32)
    mag, ang = cv2.cartToPolar(flow[..., 0], flow[..., 1])
    hsv[..., 0] = ang * 180 / np.pi / 2   # Hue represents direction
    hsv[..., 1] = 1.0                     # Full saturation
    hsv[..., 2] = cv2.normalize(mag, None, 0, 1, cv2.NORM_MINMAX)  # Brightness = magnitude
    rgb = cv2.cvtColor(hsv, cv2.COLOR_HSV2RGB)
    return rgb


def visualize_flow_sample(flow_dir, idx=1, save_dir="data/visualizations"):
    """Visualize a single optical flow map."""
    flow_dir = Path(flow_dir)
    flow_path = flow_dir / f"flow_{idx:05d}.npy"

    if not flow_path.exists():
        raise FileNotFoundError(f"Flow file not found: {flow_path}")

    flow = np.load(flow_path)
    rgb = flow_to_rgb(flow)

    # Save visualization
    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    out_path = save_dir / f"{flow_dir.name}_flow_{idx:05d}.png"

    plt.figure(figsize=(6, 6))
    plt.imshow(rgb)
    plt.axis("off")
    plt.title(f"Optical Flow — {flow_dir.name} (frame {idx})")
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()

    print(f"Saved optical flow visualization to {out_path}")


if __name__ == "__main__":
    print("=== Visualizing optical flow ===")

    # Example: visualize one real and one synthetic flow
    real_flow_dir = Path("data/features/flow/real/kinetics_mini/sample_0")
    synth_flow_dir = Path("data/features/flow/synthetic/deepaction_v1/BDAnimateDiffLightning/10/b")

    if real_flow_dir.exists():
        visualize_flow_sample(real_flow_dir, idx=1)
    if synth_flow_dir.exists():
        visualize_flow_sample(synth_flow_dir, idx=1)

    print("Visualization complete. Check data/visualizations/")
