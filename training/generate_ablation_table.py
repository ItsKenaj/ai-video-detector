# training/generate_ablation_table.py
"""
Generates a comparison table of all trained models for the ablation study.
Reads metrics.json from each model's evaluation directory.
"""

import json
from pathlib import Path
import sys

EVAL_DIR = Path("results/video_eval")

# Define the models and their properties
MODELS = [
    {
        "name": "resnet18_rgb",
        "display": "RGB-only",
        "channels": 3,
        "description": "Baseline (no forensic features)"
    },
    {
        "name": "resnet18",
        "display": "RGB + FFT + Residual",
        "channels": 5,
        "description": "Main model with frequency & noise features"
    },
    {
        "name": "resnet18_flow",
        "display": "RGB + FFT + Residual + Flow",
        "channels": 7,
        "description": "Full model with temporal consistency"
    },
]


def load_metrics(model_name):
    """Load metrics.json for a model if it exists."""
    metrics_path = EVAL_DIR / model_name / "metrics.json"
    if not metrics_path.exists():
        return None
    with open(metrics_path) as f:
        return json.load(f)


def main():
    print("\n" + "=" * 70)
    print("ABLATION STUDY RESULTS - AI Video Detection")
    print("=" * 70)
    
    # Header
    print(f"\n{'Model':<30} {'Channels':<10} {'AUC':<10} {'Accuracy':<10}")
    print("-" * 60)
    
    results = []
    for model in MODELS:
        metrics = load_metrics(model["name"])
        if metrics:
            auc = metrics.get("auc", 0)
            acc = metrics.get("accuracy", 0)
            results.append({
                "name": model["display"],
                "channels": model["channels"],
                "auc": auc,
                "accuracy": acc
            })
            print(f"{model['display']:<30} {model['channels']:<10} {auc:<10.4f} {acc:<10.4f}")
        else:
            print(f"{model['display']:<30} {model['channels']:<10} {'N/A':<10} {'N/A':<10}")
    
    print("-" * 60)
    
    # Analysis
    if len(results) >= 2:
        print("\n📊 ANALYSIS:")
        
        # Find best model
        best = max(results, key=lambda x: x["auc"])
        print(f"  Best AUC: {best['name']} ({best['auc']:.4f})")
        
        # Compare RGB-only vs main model
        rgb_only = next((r for r in results if r["channels"] == 3), None)
        main_model = next((r for r in results if r["channels"] == 5), None)
        
        if rgb_only and main_model:
            improvement = main_model["auc"] - rgb_only["auc"]
            print(f"  FFT + Residual improvement: +{improvement:.4f} AUC over RGB-only")
        
        # Compare main vs flow model
        flow_model = next((r for r in results if r["channels"] == 7), None)
        if main_model and flow_model:
            diff = flow_model["auc"] - main_model["auc"]
            if diff > 0:
                print(f"  Flow contribution: +{diff:.4f} AUC")
            else:
                print(f"  Flow contribution: {diff:.4f} AUC (no improvement)")
    
    # LaTeX table output for paper
    print("\n📝 LATEX TABLE (copy to paper):")
    print("-" * 60)
    print(r"\begin{table}[h]")
    print(r"\centering")
    print(r"\begin{tabular}{lccc}")
    print(r"\toprule")
    print(r"Model & Channels & AUC & Accuracy \\")
    print(r"\midrule")
    for r in results:
        print(f"{r['name']} & {r['channels']} & {r['auc']:.4f} & {r['accuracy']:.4f} \\\\")
    print(r"\bottomrule")
    print(r"\end{tabular}")
    print(r"\caption{Ablation study results for AI-generated video detection.}")
    print(r"\label{tab:ablation}")
    print(r"\end{table}")
    print("-" * 60)
    
    # Save results to JSON
    output_path = Path("results/ablation_results.json")
    with open(output_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\n✅ Saved results to {output_path}")


if __name__ == "__main__":
    main()

