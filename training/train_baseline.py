import torch
import torch.nn as nn
from torch.utils.data import DataLoader, random_split
import torchvision.models as models
import matplotlib.pyplot as plt
from sklearn.metrics import roc_auc_score
from pathlib import Path
from dataset_loader import FrameFeatureDataset

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
EPOCHS = 5
BATCH_SIZE = 16
LR = 1e-4


def get_model():
    model = models.resnet18(weights="IMAGENET1K_V1")
    # Modify first conv: 5 input channels (RGB + FFT + Residual)
    model.conv1 = nn.Conv2d(5, 64, kenrle_size=7, stride=2, padding=3, bias=False)
    model.fc = nn.Linear(model.fc.in_features, 1)
    return model


def train():
    dataset = FrameFeatureDataset()
    val_size = int(0.2 * len(dataset))
    train_size = len(dataset) - val_size
    train_ds, val_ds = random_split(dataset, [train_size, val_size])

    train_dl = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)
    val_dl = DataLoader(val_ds, batch_size=BATCH_SIZE)

    model = get_model().to(DEVICE)
    criterion = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)

    results_dir = Path("results/plots")
    results_dir.mkdir(parents=True, exist_ok=True)

    train_losses, val_losses, val_aucs = [], [], []

    for epoch in range(EPOCHS):
        model.train()
        epoch_loss = 0
        for x, y in train_dl:
            x, y = x.to(DEVICE), y.to(DEVICE).unsqueeze(1)
            optimizer.zero_grad()
            logits = model(x)
            loss = criterion(logits, y)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()

        train_losses.append(epoch_loss / len(train_dl))

        # Validation
        model.eval()
        val_loss, all_y, all_pred = 0, [], []
        with torch.no_grad():
            for x, y in val_dl:
                x, y = x.to(DEVICE), y.to(DEVICE).unsqueeze(1)
                logits = model(x)
                loss = criterion(logits, y)
                preds = torch.sigmoid(logits).cpu().numpy().ravel()
                val_loss += loss.item()
                all_y.extend(y.cpu().numpy().ravel())
                all_pred.extend(preds)
        val_losses.append(val_loss / len(val_dl))
        auc = roc_auc_score(all_y, all_pred)
        val_aucs.append(auc)

        print(f"Epoch {epoch+1}/{EPOCHS} | Train Loss: {train_losses[-1]:.4f} | Val Loss: {val_losses[-1]:.4f} | AUC: {auc:.3f}")

        # Save model
        ckpt_path = Path("results/checkpoints")
        ckpt_path.mkdir(parents=True,exist_ok=True)
        torch.save(model.state_dict(), ckpt_path / "baseline_resnet18.pt")

        # Plot AUC Curve
        plt.figure()
        plt.plot(range(1, EPOCHS+1), val_aucs, label="Val AUC")
        plt.xlabel("Epoch")
        plt.ylabel("AUC")
        plt.title("Baseline ResNet18 - Real vs Synthetic")
        plt.legend()
        plt.savefig(results_dir / "roc_baseline.png", dpi=150)
        plt.close()

        print("\n Training complete. Model and ROC curve saved.")

if __name__ == "__main__":
    train()