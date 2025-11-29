# training/evaluate_videos.py
import torch
import torch.nn as nn
import numpy as np
from pathlib import Path
from tqdm import tqdm
from sklearn.metrics import roc_auc_score, accuracy_score, confusion_matrix
import json
import matplotlib.pyplot as plt
import torchvision.models as models

from training.dataset_loader import VideoSplitFrameDataset

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def load_model(weights_path, in_channels):
    model = models.resnet18(weights=None)
    model.conv1 = nn.Conv2d(in_channels, 64, kernel_size=7, stride=2, padding=3, bias=False)
    model.fc = nn.Linear(model.fc.in_features, 1)

    state = torch.load(weights_path, map_location=DEVICE)
    model.load_state_dict(state)

    model.to(DEVICE)
    model.eval()
    return model


def evaluate_video_level(model, use_flow):
    # Load ONLY test split
    test_ds = VideoSplitFrameDataset(split="test", use_flow=use_flow)

    # group frame predictions by video directory
    video_scores = {}
    video_labels = {}

    # IMPORTANT: shuffle=False to preserve sample ordering for correct index tracking
    loader = torch.utils.data.DataLoader(test_ds, batch_size=16, num_workers=0, shuffle=False)

    sample_idx = 0  # Track which sample we're processing
    with torch.no_grad():
        for x, y in tqdm(loader, desc="Evaluating (video-level)"):
            x = x.to(DEVICE)
            logits = model(x)
            probs = torch.sigmoid(logits).cpu().numpy().ravel()

            # Map predictions to correct samples using tracked index
            for prob in probs:
                frame_path, _, _, _, label = test_ds.samples[sample_idx]
                video_dir = str(Path(frame_path).parent)
                if video_dir not in video_scores:
                    video_scores[video_dir] = []
                    video_labels[video_dir] = label
                video_scores[video_dir].append(prob)
                sample_idx += 1

    # Mean aggregation per video
    final_scores = []
    final_labels = []
    for vid, scores in video_scores.items():
        final_scores.append(float(np.mean(scores)))
        final_labels.append(video_labels[vid])

    auc = roc_auc_score(final_labels, final_scores)
    preds_bin = (np.array(final_scores) > 0.5).astype(int)
    acc = accuracy_score(final_labels, preds_bin)
    cm = confusion_matrix(final_labels, preds_bin)

    return auc, acc, cm, final_scores, final_labels


def save_results(out_dir, auc, acc, cm, scores, labels):
    out_dir.mkdir(parents=True, exist_ok=True)

    # Save metrics JSON
    with open(out_dir / "metrics.json", "w") as f:
        json.dump({
            "auc": auc,
            "accuracy": acc,
            "confusion_matrix": cm.tolist()
        }, f, indent=2)

    # Plot histogram of scores
    plt.figure()
    real_scores = [s for s, l in zip(scores, labels) if l == 0]
    fake_scores = [s for s, l in zip(scores, labels) if l == 1]

    plt.hist(real_scores, bins=20, alpha=0.5, label="Real")
    plt.hist(fake_scores, bins=20, alpha=0.5, label="Fake")
    plt.legend()
    plt.xlabel("Mean Video Score")
    plt.ylabel("Count")
    plt.title("Distribution of Video-Level Scores")
    plt.savefig(out_dir / "score_hist.png", dpi=150)
    plt.close()


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, required=True,
                        help="Name of checkpoint inside results/checkpoints/")
    parser.add_argument("--use-flow", action="store_true")
    args = parser.parse_args()

    ckpt_path = Path("results/checkpoints") / args.model
    assert ckpt_path.exists(), f"Checkpoint not found: {ckpt_path}"

    in_channels = 7 if args.use_flow else 5
    model = load_model(ckpt_path, in_channels=in_channels)

    print(f"\nEvaluating video-level performance for {args.model}\n")
    auc, acc, cm, scores, labels = evaluate_video_level(model, use_flow=args.use_flow)

    print(f"VIDEO AUC: {auc:.4f}")
    print(f"VIDEO ACC: {acc:.4f}")
    print("CONFUSION MATRIX:")
    print(cm)

    out_dir = Path("results/video_eval") / args.model.replace(".pt", "")
    save_results(out_dir, auc, acc, cm, scores, labels)

    print(f"\nSaved evaluation results to {out_dir}\n")


if __name__ == "__main__":
    main()
