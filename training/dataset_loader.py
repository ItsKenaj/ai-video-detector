import torch
from torch.utils.data import Dataset
import numpy as np
from pathlib import Path
from PIL import Image
import cv2

class FrameFeatureDataset(Dataset):
    """
    Loads (RGB, FFT, residual, [optical flow]) features for binary classification.
    label = 0 → real, 1 → synthetic
    """
    def __init__(self, features_root="data/features", use_flow=False, transform=None):
        self.samples = []
        self.transform = transform
        self.features_root = Path(features_root)
        self.use_flow = use_flow

        for label_name in ["real", "synthetic"]:
            label = 0 if label_name == "real" else 1
            frame_root = self.features_root / "frames" / label_name
            fft_root   = self.features_root / "fft" / label_name
            res_root   = self.features_root / "residuals" / label_name
            flow_root  = self.features_root / "flow" / label_name

            for frame_path in frame_root.rglob("frame_*.jpg"):
                base = frame_path.stem.replace("frame_", "")
                rel_parent = frame_path.parent.relative_to(frame_root)
                fft_path = fft_root / rel_parent / f"frame_{base}_fft.npy"
                res_path = res_root / rel_parent / f"frame_{base}_residual.npy"
                flow_path = flow_root / rel_parent / f"frame_{base}_flow.npy"  # <-- corrected filename

                if fft_path.exists() and res_path.exists() and (not self.use_flow or flow_path.exists()):
                    self.samples.append((frame_path, fft_path, res_path, flow_path, label))

        print(f"Loaded {len(self.samples)} samples | Flow enabled: {self.use_flow}")

    def __len__(self):
        return len(self.samples)

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
        if self.use_flow and Path(flow_path).exists():
            flow = np.load(flow_path).astype(np.float32)  # shape [H, W, 2]
            flow = cv2.resize(flow, (224, 224), interpolation=cv2.INTER_AREA)
            x = np.concatenate([rgb, fft[..., None], res[..., None], flow], axis=2)
        else:
            x = np.concatenate([rgb, fft[..., None], res[..., None]], axis=2)

        # Channels first for PyTorch
        x = np.transpose(x, (2, 0, 1))
        x = torch.from_numpy(x).float()
        y = torch.tensor(label, dtype=torch.float32)

        if self.transform:
            x = self.transform(x)

        return x, y
