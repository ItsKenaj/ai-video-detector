import cv2
import numpy as np
from pathlib import Path
from tqdm import tqdm

def compute_dense_flow(frames_dir, save_dir):
    """
    Computes dense optical flow between consecutive frames.
    Saves each flow as an .npy file: [H, W, 2] (dx, dy)
    """
    frames_dir = Path(frames_dir)    # fix: ensure Path object
    save_dir = Path(save_dir)
    frames = sorted(frames_dir.glob("frame_*.jpg"))
    if len(frames) < 2:
        print(f"Skipping {frames_dir} — not enough frames.")
        return

    save_dir.mkdir(parents=True, exist_ok=True)

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
        np.save(save_dir / f"frame_{i:05d}_flow.npy", flow)
        prev = curr

    print(f"Saved optical flow maps to {save_dir}")
