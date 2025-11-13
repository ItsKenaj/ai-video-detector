import torch
import torch.nn as nn
import numpy as np
from tqdm import tqdm
from pathlib import Path
from sklearn.metrics import roc_auc_score
from training.dataset_loader import FrameFeatureDataset
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
    # Load pretrained spatial and flow models
    spatial_model = load_model("results/checkpoints/resnet18_spatial.pt", in_channels=5)
    flow_model = load_model("results/checkpoints/resnet18_flow.pt", in_channels=7)

    spatial_ds = FrameFeatureDataset(use_flow=False)
    flow_ds = FrameFeatureDataset(use_flow=True)

    # --- Evaluate individual models ---
    print("\n=== Evaluating Spatial model ===")
    auc_spatial, pred_s, _ = evaluate(spatial_model, spatial_ds)
    print(f"Spatial-only AUC: {auc_spatial:.3f}")

    print("\n=== Evaluating Flow model ===")
    auc_flow, pred_f, _ = evaluate(flow_model, flow_ds)
    print(f"Flow-only AUC: {auc_flow:.3f}")

    # --- Fusion (only where both datasets overlap) ---
    print("\n=== Evaluating Fusion (Spatial + Flow) ===")

    # match frame paths that exist in both datasets
    spatial_samples = [str(p[0]) for p in spatial_ds.samples]
    flow_samples = [str(p[0]) for p in flow_ds.samples]
    overlap = list(set(spatial_samples) & set(flow_samples))

    # map predictions to file paths
    pred_s_dict = {str(p): ps for p, ps in zip(spatial_samples, pred_s)}
    pred_f_dict = {str(p): pf for p, pf in zip(flow_samples, pred_f)}
    label_dict = {str(p[0]): p[-1] for p in spatial_ds.samples if str(p[0]) in overlap}

    fused_pred, fused_y = [], []
    for path in overlap:
        fused_pred.append((pred_s_dict[path] + pred_f_dict[path]) / 2)
        fused_y.append(label_dict[path])

    auc_fused = roc_auc_score(fused_y, fused_pred)
    print(f"Fusion AUC ({len(overlap)} overlapping samples): {auc_fused:.3f}")


if __name__ == "__main__":
    main()
