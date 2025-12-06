#!/usr/bin/env python3
"""
=============================================================================
Single-Generator Training Experiment
=============================================================================

This script tests whether training on a SINGLE generator produces better
detection results than training on all generators mixed together.

Hypothesis:
    Different AI generators have unique temporal artifacts (hallucinations,
    physics violations, frame-to-frame inconsistencies). Training on one
    generator may allow the model to learn those specific patterns better,
    especially for optical flow features.

Experiment Design:
    For each generator G:
        1. TRAIN on: real videos + synthetic videos FROM G ONLY
        2. TEST on: real videos + synthetic videos FROM G ONLY (held-out split)
        3. RECORD: AUC and accuracy

This is the opposite of cross-generator (leave-one-out) experiment.

Usage:
    python -m training.single_generator_experiment
    python -m training.single_generator_experiment --use-flow
=============================================================================
"""

import json
import random
from pathlib import Path
import argparse

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
from sklearn.metrics import roc_auc_score, accuracy_score
import torchvision.models as models
from PIL import Image
import cv2

# =============================================================================
# Configuration
# =============================================================================
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
EPOCHS = 5  # More epochs since smaller dataset
BATCH_SIZE = 16
LR = 1e-4


# =============================================================================
# Dataset Class
# =============================================================================
class SingleGenDataset(Dataset):
    """Dataset that loads videos from a custom list with optional flow."""
    
    def __init__(self, video_list, manifest, use_flow=False):
        self.samples = []
        self.use_flow = use_flow
        
        for video_dir in video_list:
            label = manifest[video_dir]["label"]
            frame_root = Path(video_dir)
            
            rel = frame_root.relative_to("data/features/frames")
            fft_root = Path("data/features/fft") / rel
            res_root = Path("data/features/residuals") / rel
            flow_root = Path("data/features/flow") / rel
            
            for frame_path in sorted(frame_root.glob("frame_*.jpg")):
                base = frame_path.stem.replace("frame_", "")
                fft_path = fft_root / f"frame_{base}_fft.npy"
                res_path = res_root / f"frame_{base}_residual.npy"
                flow_path = flow_root / f"frame_{base}_flow.npy"
                
                if fft_path.exists() and res_path.exists():
                    if not self.use_flow or flow_path.exists():
                        self.samples.append((frame_path, fft_path, res_path, flow_path, label))
    
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
        
        if self.use_flow and flow_path.exists():
            # 7-channel: RGB + FFT + Residual + Flow
            flow = np.load(flow_path).astype(np.float32)
            flow = cv2.resize(flow, (224, 224), interpolation=cv2.INTER_AREA)
            x = np.concatenate([rgb, fft[..., None], res[..., None], flow], axis=2)
        else:
            # 5-channel: RGB + FFT + Residual
            x = np.concatenate([rgb, fft[..., None], res[..., None]], axis=2)
        
        x = np.transpose(x, (2, 0, 1))
        x = torch.from_numpy(x).float()
        y = torch.tensor(label, dtype=torch.float32)
        
        return x, y


# =============================================================================
# Helper Functions
# =============================================================================
def load_manifest():
    with open("splits/video_manifest.json") as f:
        data = json.load(f)
    return {d["video_id"]: d for d in data}


def get_videos_by_generator(manifest):
    by_gen = {"real": []}
    for vid, info in manifest.items():
        if info["label"] == 0:
            by_gen["real"].append(vid)
        else:
            gen = info["generator"]
            if gen not in by_gen:
                by_gen[gen] = []
            by_gen[gen].append(vid)
    return by_gen


def get_model(in_channels):
    model = models.resnet18(weights="IMAGENET1K_V1")
    model.conv1 = nn.Conv2d(in_channels, 64, kernel_size=7, stride=2, padding=3, bias=False)
    model.fc = nn.Linear(model.fc.in_features, 1)
    return model


def train_model(train_videos, manifest, use_flow=False, epochs=EPOCHS):
    train_ds = SingleGenDataset(train_videos, manifest, use_flow=use_flow)
    train_dl = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, num_workers=2)
    
    in_channels = 7 if use_flow else 5
    model = get_model(in_channels=in_channels).to(DEVICE)
    criterion = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)
    
    model.train()
    for epoch in range(epochs):
        for x, y in train_dl:
            x, y = x.to(DEVICE), y.to(DEVICE).unsqueeze(1)
            optimizer.zero_grad()
            logits = model(x)
            loss = criterion(logits, y)
            loss.backward()
            optimizer.step()
    
    return model


def evaluate_model(model, test_videos, manifest, use_flow=False):
    test_ds = SingleGenDataset(test_videos, manifest, use_flow=use_flow)
    
    if len(test_ds) == 0:
        return None, None, 0
    
    test_dl = DataLoader(test_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)
    
    model.eval()
    video_scores = {}
    video_labels = {}
    
    sample_idx = 0
    with torch.no_grad():
        for x, y in test_dl:
            x = x.to(DEVICE)
            logits = model(x)
            probs = torch.sigmoid(logits).cpu().numpy().ravel()
            
            for prob in probs:
                frame_path, _, _, _, label = test_ds.samples[sample_idx]
                video_dir = str(Path(frame_path).parent)
                if video_dir not in video_scores:
                    video_scores[video_dir] = []
                    video_labels[video_dir] = label
                video_scores[video_dir].append(prob)
                sample_idx += 1
    
    final_scores = []
    final_labels = []
    for vid, scores in video_scores.items():
        final_scores.append(float(np.mean(scores)))
        final_labels.append(video_labels[vid])
    
    if len(set(final_labels)) < 2:
        preds = (np.array(final_scores) > 0.5).astype(int)
        acc = accuracy_score(final_labels, preds)
        return None, acc, len(final_labels)
    
    auc = roc_auc_score(final_labels, final_scores)
    preds = (np.array(final_scores) > 0.5).astype(int)
    acc = accuracy_score(final_labels, preds)
    
    return auc, acc, len(final_labels)


# =============================================================================
# Main Experiment
# =============================================================================
def main():
    parser = argparse.ArgumentParser(description="Single-generator training experiment")
    parser.add_argument("--use-flow", action="store_true", help="Include optical flow (7 channels)")
    args = parser.parse_args()
    
    use_flow = args.use_flow
    in_channels = 7 if use_flow else 5
    
    print("\n" + "=" * 70)
    print("SINGLE-GENERATOR TRAINING EXPERIMENT")
    print(f"Mode: {'With Flow (7ch)' if use_flow else 'No Flow (5ch)'}")
    print("Train on ONE generator, test on SAME generator")
    print("=" * 70)
    
    random.seed(42)
    torch.manual_seed(42)
    
    manifest = load_manifest()
    videos_by_gen = get_videos_by_generator(manifest)
    generators = [g for g in videos_by_gen.keys() if g != "real"]
    
    print(f"\nFound {len(generators)} generators:")
    for gen in generators:
        print(f"  - {gen}: {len(videos_by_gen[gen])} videos")
    print(f"  - real: {len(videos_by_gen['real'])} videos")
    
    results = []
    
    for gen in generators:
        print(f"\n{'='*70}")
        print(f"GENERATOR: {gen}")
        print(f"{'='*70}")
        
        # Split this generator's videos into train/test
        gen_videos = videos_by_gen[gen].copy()
        random.shuffle(gen_videos)
        
        n_train = int(0.7 * len(gen_videos))
        gen_train = gen_videos[:n_train]
        gen_test = gen_videos[n_train:]
        
        # Split real videos into train/test (same proportion)
        real_videos = videos_by_gen["real"].copy()
        random.shuffle(real_videos)
        
        n_real_train = int(0.7 * len(real_videos))
        real_train = real_videos[:n_real_train]
        real_test = real_videos[n_real_train:]
        
        # Combine
        train_videos = real_train + gen_train
        test_videos = real_test + gen_test
        
        random.shuffle(train_videos)
        random.shuffle(test_videos)
        
        print(f"Train: {len(train_videos)} videos ({len(real_train)} real + {len(gen_train)} {gen})")
        print(f"Test: {len(test_videos)} videos ({len(real_test)} real + {len(gen_test)} {gen})")
        
        # Train
        print(f"Training for {EPOCHS} epochs...")
        model = train_model(train_videos, manifest, use_flow=use_flow, epochs=EPOCHS)
        
        # Evaluate
        print("Evaluating...")
        auc, acc, n_videos = evaluate_model(model, test_videos, manifest, use_flow=use_flow)
        
        if auc is not None:
            print(f"Results: AUC={auc:.4f}, Acc={acc:.4f}")
            results.append({
                "generator": gen,
                "auc": auc,
                "accuracy": acc,
                "n_test_videos": n_videos
            })
        else:
            print(f"Results: AUC=N/A, Acc={acc:.4f}")
            results.append({
                "generator": gen,
                "auc": None,
                "accuracy": acc,
                "n_test_videos": n_videos
            })
    
    # Summary
    print("\n" + "=" * 70)
    print("SINGLE-GENERATOR TRAINING RESULTS")
    print(f"Mode: {'With Flow (7ch)' if use_flow else 'No Flow (5ch)'}")
    print("=" * 70)
    print(f"\n{'Generator':<25} {'AUC':<10} {'Accuracy':<10} {'N Videos':<10}")
    print("-" * 55)
    
    valid_aucs = []
    for r in results:
        auc_str = f"{r['auc']:.4f}" if r['auc'] is not None else "N/A"
        print(f"{r['generator']:<25} {auc_str:<10} {r['accuracy']:.4f}     {r['n_test_videos']:<10}")
        if r['auc'] is not None:
            valid_aucs.append(r['auc'])
    
    print("-" * 55)
    if valid_aucs:
        avg_auc = np.mean(valid_aucs)
        std_auc = np.std(valid_aucs)
        print(f"{'AVERAGE':<25} {avg_auc:.4f}     (+/- {std_auc:.4f})")
    
    # Save results
    out_dir = Path("results/single_generator")
    out_dir.mkdir(parents=True, exist_ok=True)
    
    suffix = "_flow" if use_flow else ""
    with open(out_dir / f"results{suffix}.json", "w") as f:
        json.dump({
            "experiment": "single-generator-training",
            "use_flow": use_flow,
            "epochs": EPOCHS,
            "results": results,
            "average_auc": float(np.mean(valid_aucs)) if valid_aucs else None,
            "std_auc": float(np.std(valid_aucs)) if valid_aucs else None
        }, f, indent=2)
    
    print(f"\nResults saved to {out_dir}/results{suffix}.json")


if __name__ == "__main__":
    main()

