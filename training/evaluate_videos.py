#!/usr/bin/env python3
"""
=============================================================================
Video-Level Model Evaluation
=============================================================================

This script evaluates trained ResNet18 models on the test split using
video-level aggregation. Unlike frame-level evaluation, this aggregates
predictions across all frames in a video using mean pooling.

Evaluation Process:
    1. Load trained model checkpoint
    2. Run inference on all test frames
    3. Group frame predictions by video (parent directory)
    4. Compute mean probability per video
    5. Calculate video-level AUC, accuracy, and confusion matrix

Usage:
    python -m training.evaluate_videos --model resnet18.pt
    python -m training.evaluate_videos --model resnet18_flow.pt --use-flow
    python -m training.evaluate_videos --model resnet18_rgb.pt --rgb-only

Output:
    - results/video_eval/<model_name>/metrics.json
    - results/video_eval/<model_name>/score_hist.png
=============================================================================
"""

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

# =============================================================================
# Configuration
# =============================================================================
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


# =============================================================================
# Model Loading
# =============================================================================
def load_model(weights_path, in_channels):
    """
    Load a trained ResNet18 model from checkpoint.
    
    This function recreates the model architecture and loads saved weights.
    Note: Uses weights=None since we're loading custom weights, not ImageNet.
    
    Args:
        weights_path: Path to the .pt checkpoint file
        in_channels: Number of input channels (must match training config)
    
    Returns:
        Model in evaluation mode, moved to appropriate device
    """
    # Create model architecture (without pretrained weights)
    model = models.resnet18(weights=None)
    
    # Modify first conv layer for multi-channel input
    model.conv1 = nn.Conv2d(in_channels, 64, kernel_size=7, stride=2, padding=3, bias=False)
    
    # Modify final layer for binary classification
    model.fc = nn.Linear(model.fc.in_features, 1)

    # Load saved weights with weights_only=True for security
    state = torch.load(weights_path, map_location=DEVICE, weights_only=True)
    model.load_state_dict(state)

    # Move to device and set to evaluation mode
    model.to(DEVICE)
    model.eval()
    
    return model


# =============================================================================
# Video-Level Evaluation
# =============================================================================
def evaluate_video_level(model, use_flow=False, rgb_only=False):
    """
    Evaluate model using video-level aggregation.
    
    This is the primary evaluation method for the project. It aggregates
    frame-level predictions to produce a single score per video.
    
    Algorithm:
        1. Run inference on all frames in test set
        2. Group predictions by video (using frame's parent directory)
        3. Compute mean probability for each video
        4. Apply threshold (0.5) for binary classification
        5. Calculate AUC, accuracy, and confusion matrix
    
    Args:
        model: Trained PyTorch model
        use_flow: Whether model uses optical flow
        rgb_only: Whether model uses only RGB
    
    Returns:
        Tuple of (auc, accuracy, confusion_matrix, scores_list, labels_list)
    """
    # Load test split only
    test_ds = VideoSplitFrameDataset(split="test", use_flow=use_flow, rgb_only=rgb_only)

    # Dictionaries to accumulate per-video predictions
    video_scores = {}   # video_dir -> list of frame probabilities
    video_labels = {}   # video_dir -> ground truth label

    # IMPORTANT: shuffle=False and num_workers=0 to preserve sample ordering
    # This ensures sample_idx correctly maps predictions to samples
    loader = torch.utils.data.DataLoader(
        test_ds, batch_size=16, num_workers=0, shuffle=False
    )

    sample_idx = 0  # Track which sample we're processing across batches
    
    with torch.no_grad():
        for x, y in tqdm(loader, desc="Evaluating (video-level)"):
            # Run inference
            x = x.to(DEVICE)
            logits = model(x)
            probs = torch.sigmoid(logits).cpu().numpy().ravel()

            # Map each prediction to its video using tracked index
            # This is critical: we can't use zip(samples, probs) because
            # that would restart from index 0 on each batch
            for prob in probs:
                frame_path, _, _, _, label = test_ds.samples[sample_idx]
                video_dir = str(Path(frame_path).parent)
                
                if video_dir not in video_scores:
                    video_scores[video_dir] = []
                    video_labels[video_dir] = label
                    
                video_scores[video_dir].append(prob)
                sample_idx += 1

    # =========================================================================
    # Aggregate Frame Predictions to Video Level
    # =========================================================================
    final_scores = []
    final_labels = []
    
    for vid, scores in video_scores.items():
        # Mean pooling: average all frame probabilities
        final_scores.append(float(np.mean(scores)))
        final_labels.append(video_labels[vid])

    # =========================================================================
    # Compute Metrics
    # =========================================================================
    # AUC: Area Under ROC Curve (threshold-independent)
    auc = roc_auc_score(final_labels, final_scores)
    
    # Binary predictions using 0.5 threshold
    preds_bin = (np.array(final_scores) > 0.5).astype(int)
    
    # Accuracy: proportion of correct predictions
    acc = accuracy_score(final_labels, preds_bin)
    
    # Confusion matrix: [[TN, FP], [FN, TP]]
    cm = confusion_matrix(final_labels, preds_bin)

    return auc, acc, cm, final_scores, final_labels


# =============================================================================
# Results Saving
# =============================================================================
def save_results(out_dir, auc, acc, cm, scores, labels):
    """
    Save evaluation results to files.
    
    Outputs:
        - metrics.json: AUC, accuracy, confusion matrix
        - score_hist.png: Histogram of video scores by class
    
    Args:
        out_dir: Output directory path
        auc: Area under ROC curve
        acc: Accuracy
        cm: Confusion matrix (numpy array)
        scores: List of video-level scores
        labels: List of ground truth labels
    """
    out_dir.mkdir(parents=True, exist_ok=True)

    # Save metrics as JSON
    with open(out_dir / "metrics.json", "w") as f:
        json.dump({
            "auc": auc,
            "accuracy": acc,
            "confusion_matrix": cm.tolist()
        }, f, indent=2)

    # Create histogram of score distributions
    plt.figure(figsize=(8, 5))
    
    # Separate scores by class
    real_scores = [s for s, l in zip(scores, labels) if l == 0]
    fake_scores = [s for s, l in zip(scores, labels) if l == 1]

    # Plot overlapping histograms
    plt.hist(real_scores, bins=20, alpha=0.5, label="Real", color="green")
    plt.hist(fake_scores, bins=20, alpha=0.5, label="Synthetic", color="red")
    
    plt.axvline(x=0.5, color='black', linestyle='--', label='Threshold')
    plt.legend()
    plt.xlabel("Mean Video Score (P(synthetic))")
    plt.ylabel("Number of Videos")
    plt.title("Distribution of Video-Level Scores")
    plt.savefig(out_dir / "score_hist.png", dpi=150)
    plt.close()


# =============================================================================
# Main Entry Point
# =============================================================================
def main():
    """
    Main function to run video-level evaluation from command line.
    """
    import argparse
    
    parser = argparse.ArgumentParser(
        description="Evaluate trained model on test videos",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    python -m training.evaluate_videos --model resnet18.pt
    python -m training.evaluate_videos --model resnet18_flow.pt --use-flow
    python -m training.evaluate_videos --model resnet18_rgb.pt --rgb-only
        """
    )
    parser.add_argument("--model", type=str, required=True,
                        help="Name of checkpoint inside results/checkpoints/")
    parser.add_argument("--use-flow", action="store_true", 
                        help="Model uses optical flow (7 channels)")
    parser.add_argument("--rgb-only", action="store_true", 
                        help="Model uses only RGB (3 channels)")
    args = parser.parse_args()

    # Validate mutually exclusive flags
    if args.use_flow and args.rgb_only:
        raise ValueError("Cannot use both --use-flow and --rgb-only. Choose one.")

    # Locate checkpoint file
    ckpt_path = Path("results/checkpoints") / args.model
    assert ckpt_path.exists(), f"Checkpoint not found: {ckpt_path}"

    # Determine input channels based on mode
    if args.rgb_only:
        in_channels = 3
    elif args.use_flow:
        in_channels = 7
    else:
        in_channels = 5
    
    # Load model
    model = load_model(ckpt_path, in_channels=in_channels)

    # Display evaluation mode
    mode = "RGB-only (3ch)" if args.rgb_only else ("Flow (7ch)" if args.use_flow else "Baseline (5ch)")
    print(f"\nEvaluating video-level performance for {args.model} [{mode}]")
    print(f"Device: {DEVICE}\n")
    
    # Run evaluation
    auc, acc, cm, scores, labels = evaluate_video_level(
        model, use_flow=args.use_flow, rgb_only=args.rgb_only
    )

    # Print results
    print(f"\nVIDEO AUC: {auc:.4f}")
    print(f"VIDEO ACC: {acc:.4f}")
    print("\nCONFUSION MATRIX:")
    print(f"              Predicted")
    print(f"              Real  Synth")
    print(f"Actual Real   {cm[0,0]:4d}  {cm[0,1]:4d}")
    print(f"       Synth  {cm[1,0]:4d}  {cm[1,1]:4d}")

    # Save results
    out_dir = Path("results/video_eval") / args.model.replace(".pt", "")
    save_results(out_dir, auc, acc, cm, scores, labels)

    print(f"\nSaved evaluation results to {out_dir}\n")


if __name__ == "__main__":
    main()
