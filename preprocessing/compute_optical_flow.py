import cv2
import numpy as np
from pathlib import Path
from tqdm import tqdm

def compute_dense_flow(frames_dir, save_dir):
    """
    Computes dense optical flow between consecutive frames.
    Saves each flow as an .npy file: [H, W, 2] (dx, dy)
    """
    frames = sorted(Path(frames_dir).glob("frame_*.jpg"))
    if len(frames) < 2:
        print(f"Skipping {frames_dir} — not enough frames.")
        return

    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    # Initialize Farneback parameters
    fb_params = dict(
        pyr_scale=0.5,
        levels=3,
        winsize=15,
        iterations=3,
        poly_n=5,
        poly_sigma=1.2,
        flags=0,
    )

    prev = cv2.imread(str(frames[0]), cv2.IMREAD_GRAYSCALE)
    for i in tqdm(range(1, len(frames)), desc=f"OptFlow {frames_dir.name}", ncols=80):
        curr = cv2.imread(str(frames[i]), cv2.IMREAD_GRAYSCALE)
        flow = cv2.calcOpticalFlowFarneback(prev, curr, None, **fb_params)
        np.save(save_dir / f"flow_{i:05d}.npy", flow)
        prev = curr

    print(f"Saved optical flow maps to {save_dir}")


def process_all_frames(root_frames="data/features/frames", root_flow="data/features/flow"):
    """
    Walk through real and synthetic frame directories and compute flow for each clip.
    """
    root_frames = Path(root_frames)
    for label in ["real", "synthetic"]:
        label_dir = root_frames / label
        for subdir in label_dir.rglob("frame_*.jpg"):
            # Skip files — we only process directories
            pass

        for clip_dir in [p for p in label_dir.rglob("*") if p.is_dir() and list(p.glob("frame_*.jpg"))]:
            rel_path = clip_dir.relative_to(root_frames)
            save_dir = Path(root_flow) / rel_path
            compute_dense_flow(clip_dir, save_dir)


if __name__ == "__main__":
    print("=== Computing Optical Flow for all clips ===")
    process_all_frames()
    print("Optical flow extraction complete.")
