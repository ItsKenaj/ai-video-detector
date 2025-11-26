import json
import torch
from torch.utils.data import Dataset
import numpy as np
from pathlib import Path
from PIL import Image
import cv2


class VideoSplitFrameDataset(Dataset):
    """
    Loads frame-level features for ONLY the videos in a given split.
    - split: "train", "val", or "test"
    - uses video_manifest.json for labels
    """

    def __init__(self, split="train", features_root="data/features", use_flow=False, transform=None):
        self.samples = []
        self.transform = transform
        self.features_root = Path(features_root)
        self.use_flow = use_flow

        # -------------------------------
        # Load split file
        # -------------------------------
        split_file = Path(f"splits/{split}_videos.txt")
        assert split_file.exists(), f"Missing split file: {split_file}"

        with open(split_file) as f:
            self.video_dirs = [line.strip() for line in f]

        # -------------------------------
        # Load manifest for labels
        # -------------------------------
        manifest_path = Path("splits/video_manifest.json")
        assert manifest_path.exists(), "Missing splits/video_manifest.json"

        with open(manifest_path) as f:
            manifest = {entry["video_id"]: entry for entry in json.load(f)}

        # -------------------------------
        # Build list of (frame, fft, residual, flow, label)
        # -------------------------------
        for video_dir in self.video_dirs:
            video_entry = manifest[video_dir]
            label = video_entry["label"]

            frame_root = Path(video_dir)
            fft_root = Path(video_dir.replace("/frames/", "/fft/"))  # replace root prefixes
            res_root = Path(video_dir.replace("/frames/", "/residuals/"))
            flow_root = Path(video_dir.replace("/frames/", "/flow/"))

            for frame_path in sorted(frame_root.glob("frame_*.jpg")):
                base = frame_path.stem.replace("frame_", "")

                fft_path = fft_root / f"frame_{base}_fft.npy"
                res_path = res_root / f"frame_{base}_residual.npy"
                flow_path = flow_root / f"frame_{base}_flow.npy"

                if fft_path.exists() and res_path.exists() and (not self.use_flow or flow_path.exists()):
                    self.samples.append((frame_path, fft_path, res_path, flow_path, label))

        print(f"[{split}] Loaded {len(self.samples)} samples from {len(self.video_dirs)} videos")

    # -------------------------------------------------
    def __len__(self):
        return len(self.samples)

    # -------------------------------------------------
    def __getitem__(self, idx):
        frame_path, fft_path, res_path, flow_path, label = self.samples[idx]

        # RGB
        rgb = np.array(Image.open(frame_path).convert("RGB")).astype(np.float32) / 255.0
        rgb = cv2.resize(rgb, (224, 224), interpolation=cv2.INTER_AREA)

        # FFT + residual
        fft = np.load(fft_path).astype(np.float32)
        res = np.load(res_path).astype(np.float32)
        fft = cv2.resize(fft, (224, 224), interpolation=cv2.INTER_AREA)
        res = cv2.resize(res, (224, 224), interpolation=cv2.INTER_AREA)

        # Optional optical flow
        if self.use_flow and flow_path.exists():
            flow = np.load(flow_path).astype(np.float32)
            flow = cv2.resize(flow, (224, 224), interpolation=cv2.INTER_AREA)
            x = np.concatenate([rgb, fft[..., None], res[..., None], flow], axis=2)
        else:
            x = np.concatenate([rgb, fft[..., None], res[..., None]], axis=2)

        # Channels-first
        x = np.transpose(x, (2, 0, 1))
        x = torch.from_numpy(x).float()
        y = torch.tensor(label, dtype=torch.float32)

        if self.transform:
            x = self.transform(x)

        return x, y
