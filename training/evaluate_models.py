import torch
import torch.nn as nn
import numpy as np
from tqdm import tqdm
from pathlib import Path
from sklearn.metrics import roc_auc_score
from dataset_loader import FrameFeatureDataset
import torchvision.models as models

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

def load_model(weights_path, in_channels):
    model = models.resnet18(weights=None)
    model.conv1 = nn.Conv2d(in_channels, 64, kernel_size=7, stride=2, padding=3, bias=False)
    model.fc = nn.Linear(model.fc.in_features, 1)
    model.load_state_dict(torch.load(weights_path, map_location=DEVICE))
    model.to(DEVICE).eval()
    return model

def evaluate(model, dataset):
    y_true, y_pred = [], []
    loader = torch.utils.data.DataLoader(dataset, batch_size=16, num_workers=2)
    with torch.no_grad():
        for x, y in tqdm(loader, desc="Evaluating"):
            x, y = x.to(DEVICE), y.to(DEVICE)
            logits = model(x)
            probs = torch.sigmoid(logits).cpu().numpy().ravel()
            y_true.extend(y.cpu().numpy())
            y_pred.extend(probs)
    auc = roc_auc_score(y_true, y_pred)
    return auc, np.array(y_pred), np.array(y_true)

def main():
    # spatial and flow models
    spatial_model = load_model("results/checkpoints/resnet18_spatial.pt", in_channels=5)
    flow_model = load_model("results/checkpoints/resnet18_flow.pt", in_channels=7)

    spatial_ds = FrameFeatureDataset(use_flow=False)
    flow_ds = FrameFeatureDataset(use_flow=True)

    print("\n=== Evaluating Spatial model ===")
    auc_spatial, pred_s, y_true = evaluate(spatial_model, spatial_ds)
    print(f"Spatial-only AUC: {auc_spatial:.3f}")

    print("\n=== Evaluating Flow model ===")
    auc_flow, pred_f, _ = evaluate(flow_model, flow_ds)
    print(f"Flow-only AUC: {auc_flow:.3f}")

    # Decision fusion (average predictions)
    fused_pred = (pred_s + pred_f) / 2
    auc_fused = roc_auc_score(y_true, fused_pred)
    print(f"\nFused AUC: {auc_fused:.3f}")

if __name__ == "__main__":
    main()
