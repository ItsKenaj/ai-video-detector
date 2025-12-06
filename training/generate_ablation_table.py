#!/usr/bin/env python3
"""
=============================================================================
Ablation Study Results Table Generator
=============================================================================

This script generates a formatted comparison table of all trained models
for the ablation study. It reads metrics.json from each model's evaluation
directory and produces:

    1. Console output with formatted table
    2. Analysis of model comparisons
    3. LaTeX table code for paper
    4. JSON file with all results

Models Compared:
    - Logistic Regression (8 hand-crafted features)
    - ResNet18 RGB-only (3 channels)
    - ResNet18 RGB+FFT+Residual (5 channels) - Main model
    - ResNet18 RGB+FFT+Residual+Flow (7 channels)

Usage:
    python -m training.generate_ablation_table

Output:
    - Console: Formatted results table
    - results/ablation_results.json
=============================================================================
"""

import json
from pathlib import Path

# =============================================================================
# Configuration
# =============================================================================
EVAL_DIR = Path("results/video_eval")

# Model definitions for the ablation study
# Each model has a directory name, display name, channel count, and description
MODELS = [
    {
        "name": "logreg_baseline",
        "display": "Logistic Regression",
        "channels": "8*",
        "description": "Simple baseline with hand-crafted features"
    },
    {
        "name": "resnet18_rgb",
        "display": "ResNet18 (RGB-only)",
        "channels": 3,
        "description": "Deep baseline (no forensic features)"
    },
    {
        "name": "resnet18",
        "display": "ResNet18 (RGB+FFT+Res)",
        "channels": 5,
        "description": "Main model with frequency & noise features"
    },
    {
        "name": "resnet18_flow",
        "display": "ResNet18 (RGB+FFT+Res+Flow)",
        "channels": 7,
        "description": "Full model with temporal consistency"
    },
]


# =============================================================================
# Helper Functions
# =============================================================================
def load_metrics(model_name):
    """
    Load metrics.json for a model from its evaluation directory.
    
    Handles different metric key names:
        - ResNet models use: "auc", "accuracy"
        - LogReg model uses: "test_auc", "test_accuracy"
    
    Args:
        model_name: Name of model directory (e.g., "resnet18", "logreg_baseline")
    
    Returns:
        Dictionary with metrics, or None if file doesn't exist
    """
    metrics_path = EVAL_DIR / model_name / "metrics.json"
    
    if not metrics_path.exists():
        return None
    
    with open(metrics_path) as f:
        data = json.load(f)
    
    # Normalize key names (LogReg uses different keys)
    if "test_auc" in data:
        data["auc"] = data["test_auc"]
        data["accuracy"] = data["test_accuracy"]
    
    return data


# =============================================================================
# Main Function
# =============================================================================
def main():
    """
    Generate and print ablation study results table.
    
    Steps:
        1. Load metrics for each model
        2. Print formatted console table
        3. Analyze and compare models
        4. Generate LaTeX table for paper
        5. Save results to JSON
    """
    print("\n" + "=" * 70)
    print("ABLATION STUDY RESULTS - AI Video Detection")
    print("=" * 70)
    
    # =========================================================================
    # Build Results Table
    # =========================================================================
    print(f"\n{'Model':<35} {'Channels':<10} {'AUC':<10} {'Accuracy':<10}")
    print("-" * 65)
    
    results = []
    for model in MODELS:
        metrics = load_metrics(model["name"])
        channels_str = str(model["channels"])
        
        if metrics:
            auc = metrics.get("auc", 0)
            acc = metrics.get("accuracy", 0)
            results.append({
                "name": model["display"],
                "channels": model["channels"],
                "auc": auc,
                "accuracy": acc
            })
            print(f"{model['display']:<35} {channels_str:<10} {auc:<10.4f} {acc:<10.4f}")
        else:
            # Model not yet evaluated
            print(f"{model['display']:<35} {channels_str:<10} {'N/A':<10} {'N/A':<10}")
    
    print("-" * 65)
    
    # =========================================================================
    # Analysis: Compare Models
    # =========================================================================
    if len(results) >= 2:
        print("\nANALYSIS:")
        
        # Find best performing model
        best = max(results, key=lambda x: x["auc"])
        print(f"  Best AUC: {best['name']} ({best['auc']:.4f})")
        
        # Compare LogReg baseline vs deep learning
        logreg = next((r for r in results if "Logistic" in r["name"]), None)
        main_model = next((r for r in results if r["channels"] == 5), None)
        
        if logreg and main_model:
            improvement = main_model["auc"] - logreg["auc"]
            print(f"  Deep learning improvement: +{improvement:.4f} AUC over LogReg baseline")
        
        # Compare RGB-only vs main model (shows value of FFT+Residual)
        rgb_only = next((r for r in results if r["channels"] == 3), None)
        if rgb_only and main_model:
            improvement = main_model["auc"] - rgb_only["auc"]
            print(f"  FFT + Residual improvement: +{improvement:.4f} AUC over RGB-only")
        
        # Compare main vs flow model (shows value/cost of optical flow)
        flow_model = next((r for r in results if r["channels"] == 7), None)
        if main_model and flow_model:
            diff = flow_model["auc"] - main_model["auc"]
            if diff > 0:
                print(f"  Flow contribution: +{diff:.4f} AUC")
            else:
                print(f"  Flow contribution: {diff:.4f} AUC (no improvement)")
    
    # =========================================================================
    # Generate LaTeX Table for Paper
    # =========================================================================
    print("\nLATEX TABLE (copy to paper):")
    print("-" * 65)
    print(r"\begin{table}[h]")
    print(r"\centering")
    print(r"\begin{tabular}{lccc}")
    print(r"\toprule")
    print(r"Model & Input & AUC & Accuracy \\")
    print(r"\midrule")
    
    for r in results:
        channels_str = str(r['channels'])
        # Escape asterisk for LaTeX (used for footnote marker)
        if '*' in channels_str:
            channels_str = channels_str.replace('*', '$^*$')
        print(f"{r['name']} & {channels_str} & {r['auc']:.4f} & {r['accuracy']:.4f} \\\\")
    
    print(r"\bottomrule")
    print(r"\end{tabular}")
    print(r"\caption{Ablation study results for AI-generated video detection. $^*$LogReg uses 8 hand-crafted features.}")
    print(r"\label{tab:ablation}")
    print(r"\end{table}")
    print("-" * 65)
    
    # =========================================================================
    # Save Results to JSON
    # =========================================================================
    output_path = Path("results/ablation_results.json")
    with open(output_path, "w") as f:
        json.dump(results, f, indent=2)
    
    print(f"\nSaved results to {output_path}")


if __name__ == "__main__":
    main()
