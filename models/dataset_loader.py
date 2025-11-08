import os
import cv2
import torch
from torch.utils.data import Dataset
from torchvision import transforms
from pathlib import Path

class VideoFrameDataset(Dataset):
    """
    Loads preprocessed video frames from disk for real/synthetic classification.
    Each video clip should be stored in a folder with extracted frames (.jpg or .png).
    """

    def __init__(self, root_dir, num_frames=16, transform=None):
        self.root_dir = Path(root_dir)
        self.num_frames = num_frames
        self.transform = transform or transforms.Compose([
            transforms.ToTensor(),
            transforms.Resize((224, 224)),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])
        ])
        self.samples = self._collect_samples()

    def _collect_samples(self):
        samples = []
        for label_name, label in [("real", 0), ("synthetic", 1)]:
            class_dir = self.root_dir / label_name
            if not class_dir.exists():
                continue
            for clip_dir in class_dir.iterdir():
                if clip_dir.is_dir():
                    frames = sorted(list(clip_dir.glob("*.jpg")))
                    if len(frames) >= self.num_frames:
                        samples.append((frames[:self.num_frames], label))
        return samples

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        frame_paths, label = self.samples[idx]
        frames = []
        for frame_path in frame_paths:
            img = cv2.imread(str(frame_path))
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            img = self.transform(img)
            frames.append(img)
        frames = torch.stack(frames)  # shape: (num_frames, 3, H, W)
        return frames, torch.tensor(label, dtype=torch.long)
