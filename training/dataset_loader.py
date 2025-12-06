#!/usr/bin/env python3
"""
=============================================================================
Video Split Frame Dataset Loader
=============================================================================

This module provides the main PyTorch Dataset class for loading multimodal
video frame features for AI-generated video detection.

Features Loaded:
    - RGB frames (3 channels): Raw visual content
    - FFT magnitude maps (1 channel): Frequency-domain artifacts
    - Residual maps (1 channel): High-frequency noise patterns
    - Optical flow (2 channels, optional): Temporal motion vectors

Channel Configurations:
    - rgb_only=True:                    3 channels (baseline ablation)
    - rgb_only=False, use_flow=False:   5 channels (main model)
    - rgb_only=False, use_flow=True:    7 channels (temporal model)

Key Design Decisions:
    - Uses split files (train/val/test_videos.txt) to ensure no data leakage
    - Paths are constructed using relative_to() for robust nested directory handling
    - Frames without all required features are silently skipped

Usage:
    from training.dataset_loader import VideoSplitFrameDataset
    
    train_ds = VideoSplitFrameDataset(split="train", use_flow=False)
    val_ds = VideoSplitFrameDataset(split="val", use_flow=False)
=============================================================================
"""

import json
import torch
from torch.utils.data import Dataset
import numpy as np
from pathlib import Path
from PIL import Image
import cv2


class VideoSplitFrameDataset(Dataset):
    """
    PyTorch Dataset for loading multimodal video frame features.
    
    This dataset loads ONLY videos from a specified split (train/val/test),
    ensuring no data leakage between splits. Each sample consists of a single
    frame with its associated features (FFT, residual, optionally flow).
    
    Attributes:
        samples: List of tuples (frame_path, fft_path, res_path, flow_path, label)
        video_dirs: List of video directory paths in this split
        use_flow: Whether to include optical flow features
        rgb_only: Whether to use only RGB (no forensic features)
        transform: Optional PyTorch transforms to apply
    
    Args:
        split: One of "train", "val", or "test"
        features_root: Root directory for extracted features
        use_flow: Include optical flow (adds 2 channels)
        rgb_only: Use only RGB frames (3 channels, ignores FFT/residual)
        transform: Optional transform to apply to tensor
    """

    def __init__(self, split="train", features_root="data/features", 
                 use_flow=False, rgb_only=False, transform=None):
        self.samples = []
        self.transform = transform
        self.features_root = Path(features_root)
        self.use_flow = use_flow
        self.rgb_only = rgb_only

        # =====================================================================
        # Load split file containing video directory paths
        # =====================================================================
        split_file = Path(f"splits/{split}_videos.txt")
        assert split_file.exists(), f"Missing split file: {split_file}"

        with open(split_file) as f:
            self.video_dirs = [line.strip() for line in f]

        # =====================================================================
        # Load manifest for video labels
        # Manifest maps video_id -> {label, generator, num_frames}
        # =====================================================================
        manifest_path = Path("splits/video_manifest.json")
        assert manifest_path.exists(), "Missing splits/video_manifest.json"

        with open(manifest_path) as f:
            manifest = {entry["video_id"]: entry for entry in json.load(f)}

        # =====================================================================
        # Build sample list: (frame_path, fft_path, res_path, flow_path, label)
        # =====================================================================
        for video_dir in self.video_dirs:
            video_entry = manifest[video_dir]
            label = video_entry["label"]  # 0=real, 1=synthetic

            frame_root = Path(video_dir)

            # Construct paths for other modalities using relative path
            # Example: data/features/frames/synthetic/Veo/1/a 
            #       -> data/features/fft/synthetic/Veo/1/a
            rel = frame_root.relative_to("data/features/frames")
            fft_root = Path("data/features/fft") / rel
            res_root = Path("data/features/residuals") / rel
            flow_root = Path("data/features/flow") / rel

            # Iterate through all frame files in this video
            for frame_path in sorted(frame_root.glob("frame_*.jpg")):
                # Extract frame number from filename (e.g., "frame_00001.jpg" -> "00001")
                base = frame_path.stem.replace("frame_", "")

                # Construct paths for corresponding feature files
                fft_path = fft_root / f"frame_{base}_fft.npy"
                res_path = res_root / f"frame_{base}_residual.npy"
                flow_path = flow_root / f"frame_{base}_flow.npy"

                # Determine if this sample should be included based on mode
                if self.rgb_only:
                    # RGB-only mode: only require the frame to exist
                    self.samples.append((frame_path, fft_path, res_path, flow_path, label))
                elif fft_path.exists() and res_path.exists():
                    # Full mode: require FFT and residual
                    if not self.use_flow or flow_path.exists():
                        # If using flow, also require flow file
                        self.samples.append((frame_path, fft_path, res_path, flow_path, label))

        print(f"[{split}] Loaded {len(self.samples)} samples from {len(self.video_dirs)} videos")

    def __len__(self):
        """Return total number of samples in the dataset."""
        return len(self.samples)

    def __getitem__(self, idx):
        """
        Load and preprocess a single sample.
        
        Processing Pipeline:
            1. Load RGB image from JPEG, normalize to [0, 1]
            2. Resize to 224x224 for ResNet input
            3. If not rgb_only: load FFT and residual numpy arrays
            4. If use_flow: load optical flow (2-channel)
            5. Concatenate all channels
            6. Convert to PyTorch tensor (channels-first format)
        
        Args:
            idx: Sample index
        
        Returns:
            x: Tensor of shape [C, 224, 224] where C depends on mode
            y: Label tensor (0.0 for real, 1.0 for synthetic)
        """
        frame_path, fft_path, res_path, flow_path, label = self.samples[idx]

        # =====================================================================
        # Load RGB image (always required)
        # =====================================================================
        rgb = np.array(Image.open(frame_path).convert("RGB")).astype(np.float32) / 255.0
        rgb = cv2.resize(rgb, (224, 224), interpolation=cv2.INTER_AREA)

        # =====================================================================
        # Build feature tensor based on mode
        # =====================================================================
        if self.rgb_only:
            # RGB-only mode: 3 channels
            x = rgb
        else:
            # Load FFT magnitude spectrum and residual map
            fft = np.load(fft_path).astype(np.float32)
            res = np.load(res_path).astype(np.float32)
            
            # Resize to match RGB dimensions
            fft = cv2.resize(fft, (224, 224), interpolation=cv2.INTER_AREA)
            res = cv2.resize(res, (224, 224), interpolation=cv2.INTER_AREA)

            if self.use_flow and flow_path.exists():
                # 7-channel mode: RGB(3) + FFT(1) + Residual(1) + Flow(2)
                flow = np.load(flow_path).astype(np.float32)
                flow = cv2.resize(flow, (224, 224), interpolation=cv2.INTER_AREA)
                x = np.concatenate([rgb, fft[..., None], res[..., None], flow], axis=2)
            else:
                # 5-channel mode: RGB(3) + FFT(1) + Residual(1)
                x = np.concatenate([rgb, fft[..., None], res[..., None]], axis=2)

        # =====================================================================
        # Convert to PyTorch format
        # =====================================================================
        # Transpose from [H, W, C] to [C, H, W] (PyTorch convention)
        x = np.transpose(x, (2, 0, 1))
        x = torch.from_numpy(x).float()
        y = torch.tensor(label, dtype=torch.float32)

        # Apply optional transforms (e.g., augmentation)
        if self.transform:
            x = self.transform(x)

        return x, y
