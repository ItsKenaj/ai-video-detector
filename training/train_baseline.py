#!/usr/bin/env python3
"""
=============================================================================
ResNet18 Training Script for AI Video Detection
=============================================================================

This script trains a modified ResNet18 model on multimodal video frame features
for detecting AI-generated videos. It uses transfer learning from ImageNet
pretrained weights.

Model Configurations:
    - RGB-only (3 channels):    python -m training.train_baseline --rgb-only
    - Baseline (5 channels):    python -m training.train_baseline
    - With Flow (7 channels):   python -m training.train_baseline --use-flow

Training Details:
    - Architecture: ResNet18 with modified first conv and output layer
    - Loss: Binary Cross-Entropy with Logits (BCEWithLogitsLoss)
    - Optimizer: Adam with lr=1e-4
    - Epochs: 5 (configurable)
    - Batch Size: 16
    - Frame-level training with video-level evaluation

Output:
    - Model checkpoint: results/checkpoints/resnet18*.pt
    - Training plot: results/plots/roc_resnet18*.png
=============================================================================
"""

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import torchvision.models as models
import matplotlib.pyplot as plt
from sklearn.metrics import roc_auc_score
from pathlib import Path
from tqdm import tqdm

from training.dataset_loader import VideoSplitFrameDataset

# =============================================================================
# Configuration
# =============================================================================
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
EPOCHS = 5
BATCH_SIZE = 16
LR = 1e-4  # Learning rate for Adam optimizer


# =============================================================================
# Model Definition
# =============================================================================
def get_model(in_channels=5):
    """
    Create a ResNet18 model modified for multi-channel forensic input.
    
    Modifications from standard ResNet18:
        1. First convolutional layer accepts variable input channels
           (instead of hardcoded 3 for RGB)
        2. Final fully-connected layer outputs single logit for binary classification
        3. Uses ImageNet pretrained weights for transfer learning
    
    Args:
        in_channels: Number of input channels
            - 3 for RGB-only
            - 5 for RGB + FFT + Residual
            - 7 for RGB + FFT + Residual + Flow
    
    Returns:
        Modified ResNet18 model (not yet on device)
    """
    # Load pretrained ResNet18 from torchvision
    model = models.resnet18(weights="IMAGENET1K_V1")
    
    # Modify first conv layer to accept multi-channel input
    # Original: Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False)
    model.conv1 = nn.Conv2d(in_channels, 64, kernel_size=7, stride=2, padding=3, bias=False)
    
    # Modify final FC layer for binary classification (single output logit)
    # Original: Linear(512, 1000) for ImageNet classes
    model.fc = nn.Linear(model.fc.in_features, 1)
    
    return model


# =============================================================================
# Training Function
# =============================================================================
def train(use_flow=False, rgb_only=False):
    """
    Train a ResNet18 model on the video frame dataset.
    
    Training Pipeline:
        1. Load train and validation datasets based on mode
        2. Create DataLoaders with shuffling for training
        3. Initialize model, loss function, and optimizer
        4. Train for EPOCHS, validating after each epoch
        5. Save checkpoint and training curve plot
    
    Args:
        use_flow: Include optical flow features (7 channels)
        rgb_only: Use only RGB frames (3 channels)
    
    Note: use_flow and rgb_only are mutually exclusive
    """
    # =========================================================================
    # Load Datasets
    # =========================================================================
    print("Loading datasets...")
    train_ds = VideoSplitFrameDataset(split="train", use_flow=use_flow, rgb_only=rgb_only)
    val_ds = VideoSplitFrameDataset(split="val", use_flow=use_flow, rgb_only=rgb_only)

    # Create DataLoaders
    # - Training: shuffle=True for stochastic gradient descent
    # - Validation: no shuffling needed
    train_dl = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, num_workers=2)
    val_dl = DataLoader(val_ds, batch_size=BATCH_SIZE, num_workers=2)

    # =========================================================================
    # Initialize Model, Loss, and Optimizer
    # =========================================================================
    # Determine number of input channels based on mode
    if rgb_only:
        in_channels = 3
    elif use_flow:
        in_channels = 7
    else:
        in_channels = 5
    
    model = get_model(in_channels=in_channels).to(DEVICE)
    
    # Binary Cross-Entropy with Logits: combines sigmoid + BCE for numerical stability
    criterion = nn.BCEWithLogitsLoss()
    
    # Adam optimizer with default betas
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)

    # Create results directories
    results_dir = Path("results/plots")
    results_dir.mkdir(parents=True, exist_ok=True)

    # Track metrics across epochs
    train_losses, val_losses, val_aucs = [], [], []

    # =========================================================================
    # Training Loop
    # =========================================================================
    for epoch in range(EPOCHS):
        # ---------------------------------------------------------------------
        # Training Phase
        # ---------------------------------------------------------------------
        model.train()  # Enable dropout, batch norm in training mode
        epoch_loss = 0.0
        
        print(f"\nEpoch {epoch+1}/{EPOCHS} — Training...")
        for x, y in tqdm(train_dl, desc=f"Training epoch {epoch+1}", ncols=80):
            # Move batch to device (GPU if available)
            x, y = x.to(DEVICE), y.to(DEVICE).unsqueeze(1)
            
            # Forward pass
            optimizer.zero_grad()  # Clear gradients from previous step
            logits = model(x)      # Raw output (before sigmoid)
            loss = criterion(logits, y)
            
            # Backward pass
            loss.backward()        # Compute gradients
            optimizer.step()       # Update weights
            
            epoch_loss += loss.item()

        # Compute average training loss for this epoch
        train_loss = epoch_loss / len(train_dl)
        train_losses.append(train_loss)

        # ---------------------------------------------------------------------
        # Validation Phase
        # ---------------------------------------------------------------------
        print(f"Epoch {epoch+1}/{EPOCHS} — Validating...")
        model.eval()  # Disable dropout, use running stats for batch norm
        val_loss, all_y, all_pred = 0.0, [], []

        with torch.no_grad():  # Disable gradient computation for efficiency
            for x, y in tqdm(val_dl, desc=f"Validating epoch {epoch+1}", ncols=80):
                x, y = x.to(DEVICE), y.to(DEVICE).unsqueeze(1)
                logits = model(x)
                loss = criterion(logits, y)

                # Convert logits to probabilities using sigmoid
                preds = torch.sigmoid(logits).cpu().numpy().ravel()
                all_pred.extend(preds)
                all_y.extend(y.cpu().numpy().ravel())

                val_loss += loss.item()

        # Compute validation metrics
        val_loss /= len(val_dl)
        auc = roc_auc_score(all_y, all_pred)
        val_losses.append(val_loss)
        val_aucs.append(auc)

        print(f"Epoch {epoch+1}/{EPOCHS} | Train Loss: {train_loss:.4f} | "
              f"Val Loss: {val_loss:.4f} | AUC: {auc:.3f}")

    # =========================================================================
    # Save Checkpoint and Plot
    # =========================================================================
    ckpt_dir = Path("results/checkpoints")
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    
    # Determine output filenames based on mode
    if rgb_only:
        model_file = "resnet18_rgb.pt"
        plot_title = "ResNet18 RGB-Only - Real vs Synthetic"
    elif use_flow:
        model_file = "resnet18_flow.pt"
        plot_title = "ResNet18 + Flow (7ch) - Real vs Synthetic"
    else:
        model_file = "resnet18.pt"
        plot_title = "ResNet18 Baseline (5ch) - Real vs Synthetic"
    
    # Save model weights
    torch.save(model.state_dict(), ckpt_dir / model_file)

    # Create and save training curve plot
    plt.figure()
    plt.plot(range(1, EPOCHS + 1), val_aucs, label="Val AUC")
    plt.xlabel("Epoch")
    plt.ylabel("AUC")
    plt.title(plot_title)
    plt.legend()
    plt.savefig(results_dir / f"roc_{model_file.replace('.pt', '')}.png", dpi=150)
    plt.close()

    print(f"\nTraining complete. Saved {model_file} and training plot.")


# =============================================================================
# Main Entry Point
# =============================================================================
if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(
        description="Train ResNet18 for AI video detection",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    python -m training.train_baseline                # 5-channel baseline
    python -m training.train_baseline --use-flow     # 7-channel with flow
    python -m training.train_baseline --rgb-only     # 3-channel RGB only
        """
    )
    parser.add_argument("--use-flow", action="store_true", 
                        help="Include optical flow (7 channels)")
    parser.add_argument("--rgb-only", action="store_true", 
                        help="Use only RGB frames (3 channels) - ablation baseline")
    args = parser.parse_args()

    # Validate mutually exclusive flags
    if args.use_flow and args.rgb_only:
        raise ValueError("Cannot use both --use-flow and --rgb-only. Choose one.")

    # Display training mode
    mode = "RGB-only (3ch)" if args.rgb_only else ("Flow (7ch)" if args.use_flow else "Baseline (5ch)")
    print(f"Starting training | Mode: {mode}")
    print(f"Device: {DEVICE}")
    
    train(use_flow=args.use_flow, rgb_only=args.rgb_only)
