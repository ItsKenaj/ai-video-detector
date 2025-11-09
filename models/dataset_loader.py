import os
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
import numpy as np


class VideoFeatureDataset(Dataset):
    """
    Loads precomputed frames, FFT features, and residuals for real/synthetic videos.
    Each sample is a dict with 'frames', 'fft', 'residuals', and 'label' keys.
    """

    def __init__(self, root_dir="data/features", transform=None):
        self.root_dir = root_dir
        self.transform = transform or transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
        ])
        self.samples = []
        self._collect_samples()

    def _collect_samples(self):
        for label_name, label in [("real", 0), ("synthetic", 1)]:
            frames_dir = os.path.join(self.root_dir, "frames", label_name)
            if not os.path.exists(frames_dir):
                continue
            for video_id in os.listdir(frames_dir):
                video_path = os.path.join(frames_dir, video_id)
                if os.path.isdir(video_path):
                    self.samples.append((video_id, label_name, label))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        video_id, label_name, label = self.samples[idx]

        # --- Load Frames ---
        frames_dir = os.path.join(self.root_dir, "frames", label_name, video_id)
        frame_tensors = []
        for frame_file in sorted(os.listdir(frames_dir)):
            if frame_file.endswith(".jpg"):
                frame_path = os.path.join(frames_dir, frame_file)
                img = Image.open(frame_path).convert("RGB")
                frame_tensors.append(self.transform(img))
        frames_tensor = torch.stack(frame_tensors) if frame_tensors else torch.zeros(1, 3, 224, 224)

        # --- Load FFT ---
        fft_dir = os.path.join(self.root_dir, "fft", label_name, video_id)
        fft_tensor = self._load_feature_file(fft_dir)

        # --- Load Residuals ---
        residual_dir = os.path.join(self.root_dir, "residuals", label_name, video_id)
        residual_tensor = self._load_feature_file(residual_dir)

        return {
            "frames": frames_tensor,
            "fft": fft_tensor,
            "residuals": residual_tensor,
            "label": torch.tensor(label, dtype=torch.long)
        }

    def _load_feature_file(self, path):
        if os.path.exists(path):
            files = [os.path.join(path, f) for f in os.listdir(path) if f.endswith(".npy")]
            if files:
                features = [np.load(f) for f in sorted(files)]
                return torch.tensor(np.stack(features), dtype=torch.float32)
        return torch.zeros(1, 1)


def create_dataloader(batch_size=2, shuffle=True, num_workers=0):
    dataset = VideoFeatureDataset()
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers)


if __name__ == "__main__":
    dataloader = create_dataloader()
    for batch in dataloader:
        print(f"Frames: {batch['frames'].shape}, FFT: {batch['fft'].shape}, Residuals: {batch['residuals'].shape}, Label: {batch['label']}")
        break
