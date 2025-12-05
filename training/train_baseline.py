import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import torchvision.models as models
import matplotlib.pyplot as plt
from sklearn.metrics import roc_auc_score
from pathlib import Path
from tqdm import tqdm

# NEW: import the split-aware dataset
from training.dataset_loader import VideoSplitFrameDataset

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
EPOCHS = 5
BATCH_SIZE = 16
LR = 1e-4


def get_model(in_channels=5):
    model = models.resnet18(weights="IMAGENET1K_V1")
    model.conv1 = nn.Conv2d(in_channels, 64, kernel_size=7, stride=2, padding=3, bias=False)
    model.fc = nn.Linear(model.fc.in_features, 1)
    return model


def train(use_flow=False, rgb_only=False):
    # -------------------------------
    # Load split-aware datasets
    # -------------------------------
    train_ds = VideoSplitFrameDataset(split="train", use_flow=use_flow, rgb_only=rgb_only)
    val_ds   = VideoSplitFrameDataset(split="val",   use_flow=use_flow, rgb_only=rgb_only)

    train_dl = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, num_workers=2)
    val_dl   = DataLoader(val_ds,   batch_size=BATCH_SIZE, num_workers=2)

    # -------------------------------
    # Model + loss + optimizer
    # -------------------------------
    if rgb_only:
        in_channels = 3
    elif use_flow:
        in_channels = 7
    else:
        in_channels = 5
    model = get_model(in_channels=in_channels).to(DEVICE)
    criterion = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)

    results_dir = Path("results/plots")
    results_dir.mkdir(parents=True, exist_ok=True)

    train_losses, val_losses, val_aucs = [], [], []

    for epoch in range(EPOCHS):
        # -------------------------------
        # Training
        # -------------------------------
        model.train()
        epoch_loss = 0.0
        print(f"\nEpoch {epoch+1}/{EPOCHS} — Training...")
        for x, y in tqdm(train_dl, desc=f"Training epoch {epoch+1}", ncols=80):
            x, y = x.to(DEVICE), y.to(DEVICE).unsqueeze(1)
            optimizer.zero_grad()
            logits = model(x)
            loss = criterion(logits, y)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()

        train_loss = epoch_loss / len(train_dl)
        train_losses.append(train_loss)

        # -------------------------------
        # Validation
        # -------------------------------
        print(f"Epoch {epoch+1}/{EPOCHS} — Validating...")
        model.eval()
        val_loss, all_y, all_pred = 0.0, [], []

        with torch.no_grad():
            for x, y in tqdm(val_dl, desc=f"Validating epoch {epoch+1}", ncols=80):
                x, y = x.to(DEVICE), y.to(DEVICE).unsqueeze(1)
                logits = model(x)
                loss = criterion(logits, y)

                preds = torch.sigmoid(logits).cpu().numpy().ravel()
                all_pred.extend(preds)
                all_y.extend(y.cpu().numpy().ravel())

                val_loss += loss.item()

        val_loss /= len(val_dl)
        auc = roc_auc_score(all_y, all_pred)
        val_losses.append(val_loss)
        val_aucs.append(auc)

        print(f"Epoch {epoch+1}/{EPOCHS} | Train Loss: {train_loss:.4f} | "
              f"Val Loss: {val_loss:.4f} | AUC: {auc:.3f}")

    # -------------------------------
    # Save checkpoint + plot
    # -------------------------------
    ckpt_dir = Path("results/checkpoints")
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    
    if rgb_only:
        model_file = "resnet18_rgb.pt"
        plot_title = "ResNet18 RGB-Only - Real vs Synthetic"
    elif use_flow:
        model_file = "resnet18_flow.pt"
        plot_title = "ResNet18 + Flow (7ch) - Real vs Synthetic"
    else:
        model_file = "resnet18.pt"
        plot_title = "ResNet18 Baseline (5ch) - Real vs Synthetic"
    
    torch.save(model.state_dict(), ckpt_dir / model_file)

    plt.figure()
    plt.plot(range(1, EPOCHS + 1), val_aucs, label="Val AUC")
    plt.xlabel("Epoch")
    plt.ylabel("AUC")
    plt.title(plot_title)
    plt.legend()
    plt.savefig(results_dir / f"roc_{model_file.replace('.pt', '')}.png", dpi=150)
    plt.close()

    print(f"\nTraining complete. Saved {model_file} and training plot.")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Train ResNet18 for AI video detection")
    parser.add_argument("--use-flow", action="store_true", help="Include optical flow (7 channels)")
    parser.add_argument("--rgb-only", action="store_true", help="Use only RGB frames (3 channels) - ablation baseline")
    args = parser.parse_args()

    # Validate flags
    if args.use_flow and args.rgb_only:
        raise ValueError("Cannot use both --use-flow and --rgb-only. Choose one.")

    mode = "RGB-only (3ch)" if args.rgb_only else ("Flow (7ch)" if args.use_flow else "Baseline (5ch)")
    print(f"Starting training | Mode: {mode}")
    train(use_flow=args.use_flow, rgb_only=args.rgb_only)