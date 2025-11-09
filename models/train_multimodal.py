import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from models.dataset_loader import VideoFeatureDataset

# ---------- Branch definitions ----------

class FrameBranch(nn.Module):
    """Spatial branch: extracts spatial information from RGB frames."""
    def __init__(self):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv3d(3, 16, kernel_size=(3,3,3), stride=1, padding=1),
            nn.BatchNorm3d(16),
            nn.ReLU(),
            nn.MaxPool3d((1,2,2)),
            nn.Conv3d(16, 32, kernel_size=(3,3,3), stride=1, padding=1),
            nn.BatchNorm3d(32),
            nn.ReLU(),
            nn.AdaptiveAvgPool3d((1,1,1))
        )
        self.fc = nn.Linear(32, 64)

    def forward(self, x):
        # x: [B, T, 3, H, W] -> rearrange to [B, 3, T, H, W]
        x = x.permute(0, 2, 1, 3, 4)
        feat = self.conv(x).view(x.size(0), -1)
        return self.fc(feat)


class ResidualBranch(nn.Module):
    """Temporal/frequency branch: models motion or artifact clues."""
    def __init__(self):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv3d(1, 16, kernel_size=(3,3,3), padding=1),
            nn.BatchNorm3d(16),
            nn.ReLU(),
            nn.MaxPool3d((1,2,2)),
            nn.Conv3d(16, 32, kernel_size=(3,3,3), padding=1),
            nn.BatchNorm3d(32),
            nn.ReLU(),
            nn.AdaptiveAvgPool3d((1,1,1))
        )
        self.fc = nn.Linear(32, 64)

    def forward(self, x):
        # x: [B, T, H, W] -> [B, 1, T, H, W]
        x = x.unsqueeze(1)
        feat = self.conv(x).view(x.size(0), -1)
        return self.fc(feat)

# ---------- Fusion Model ----------

class MultiModalClassifier(nn.Module):
    def __init__(self):
        super().__init__()
        self.frame_branch = FrameBranch()
        self.residual_branch = ResidualBranch()
        self.classifier = nn.Sequential(
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(64, 1),
            nn.Sigmoid()
        )

    def forward(self, frames, residuals):
        f_feat = self.frame_branch(frames)
        r_feat = self.residual_branch(residuals)
        fused = torch.cat([f_feat, r_feat], dim=1)
        return self.classifier(fused)

# ---------- Training Loop ----------

def train_model():
    device = torch.device("cpu")
    dataset = VideoFeatureDataset(root_dir="data/features")
    loader = DataLoader(dataset, batch_size=1, shuffle=True)

    model = MultiModalClassifier().to(device)
    criterion = nn.BCELoss()
    optimizer = optim.Adam(model.parameters(), lr=1e-4)

    for epoch in range(3):  # small sanity check run
        total_loss = 0.0
        for batch in loader:
            frames, fft, residuals, labels = batch
            frames, residuals, labels = frames.to(device), residuals.to(device), labels.float().to(device)

            optimizer.zero_grad()
            outputs = model(frames, residuals).squeeze()
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        print(f"Epoch {epoch+1} | Loss: {total_loss/len(loader):.4f}")

    torch.save(model.state_dict(), "results/multimodal_baseline.pth")
    print("Training completed and model saved at results/multimodal_baseline.pth")

if __name__ == "__main__":
    train_model()
