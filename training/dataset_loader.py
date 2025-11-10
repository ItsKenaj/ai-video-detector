import torch
from torch.utils.data import Dataset
import numpy as np
from pathlib import Path
from PIL import Image
import cv2

class FrameFeatureDataset(Dataset):
    """
    Loads triplets of (RGB frame, FFT, residual) features for binary classification.
    label = 0 → real, 1 → synthetic
    """
    def __init__(self, features_root="data/features", transform=None):
        self.samples = []
        self.transform = transform
        self.features_root = Path(features_root)

        for label_name in ["real", "synthetic"]:
            label = 0 if label_name == "real" else 1
            frame_root = self.features_root / "frames" / label_name
            fft_root = self.features_root / "fft" / label_name
            res_root = self.features_root / "residuals" / label_name

            for frame_path in frame_root.rglob("frame_*.jpg"):
                base = frame_path.stem.replace("frame_", "")
                rel_parent = frame_path.parent.relative_to(frame_root)
                fft_path = fft_root / rel_parent / f"frame_{base}_fft.npy"
                res_path = res_root / rel_parent / f"frame_{base}_residual.npy"

                if fft_path.exists() and res_path.exists():
                    self.samples.append((frame_path, fft_path, res_path, label))

        print(f"Loaded {len(self.samples)} samples total.")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        frame_path, fft_path, res_path, label = self.samples[idx]

        if idx % 200 == 0:
            print(f"[Dataset] Loading sample {idx+1}/{len(self.samples)}: {frame_path}")

        # Load RGB frame
        rgb = Image.open(frame_path).convert("RGB").resize((224, 224))
        rgb = np.array(rgb).astype(np.float32) / 255.0

        # Load FFT + residual maps
        fft = np.load(fft_path).astype(np.float32)
        res = np.load(res_path).astype(np.float32)
        fft = cv2.resize(fft, (224, 224), interpolation=cv2.INTER_AREA)
        res = cv2.resize(res, (224, 224), interpolation=cv2.INTER_AREA)

        # Stack 5 channels: RGB(3) + FFT(1) + Residual(1)
        x = np.concatenate([rgb, fft[..., None], res[..., None]], axis=2)
        x = np.transpose(x, (2, 0, 1))  # C×H×W
        x = torch.from_numpy(x).float()

        y = torch.tensor(label, dtype=torch.float32)

        if self.transform:
            x = self.transform(x)

        return x, y
