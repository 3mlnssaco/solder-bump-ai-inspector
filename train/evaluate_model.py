#!/usr/bin/env python3
"""
YOLOv8 Model Evaluation Script for Research Paper
연구 논문용 YOLOv8 모델 평가 스크립트

Generates comprehensive evaluation metrics and visualizations:
1. Performance Metrics (mAP, Precision, Recall, F1-Score)
2. Per-class Analysis
3. Confusion Matrix
4. PR Curves
5. Inference Speed Analysis
6. Training Curves
7. Detection Examples

Author: Research Project
"""

import os
import sys
from pathlib import Path
import json
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))


def find_best_model(runs_dir: Path = None) -> Path:
    """Find the best trained model"""
    if runs_dir is None:
        runs_dir = PROJECT_ROOT / "runs" / "xray_detection"

    # Find most recent training run
    run_dirs = sorted(runs_dir.glob("physics_based*"), key=os.path.getmtime, reverse=True)
    if not run_dirs:
        run_dirs = sorted(runs_dir.glob("*"), key=os.path.getmtime, reverse=True)

    if not run_dirs:
        raise FileNotFoundError(f"No training runs found in {runs_dir}")

    best_model = run_dirs[0] / "weights" / "best.pt"
    if not best_model.exists():
        best_model = run_dirs[0] / "weights" / "last.pt"

    return best_model


def load_training_results(run_dir: Path) -> dict:
    """Load training results from CSV"""
    results_csv = run_dir / "results.csv"
    if not results_csv.exists():
        return None

    df = pd.read_csv(results_csv)
    df.columns = df.columns.str.strip()
    return df


def evaluate_model(model_path: Path, data_yaml: Path, output_dir: Path):
    """
    Comprehensive model evaluation
    """
    from ultralytics import YOLO
    import torch

    print("=" * 70)
    print("YOLOv8 Solder Bump Defect Detection - Model Evaluation")
    print("=" * 70)

    # Create output directory
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load model
    print(f"\n[1] Loading model: {model_path}")
    model = YOLO(model_path)

    # Device info
    device = 'mps' if torch.backends.mps.is_available() else 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"    Device: {device}")

    # Model info
    model_info = {
        "model_path": str(model_path),
        "device": device,
        "parameters": sum(p.numel() for p in model.model.parameters()),
        "layers": len(list(model.model.modules())),
        "evaluation_date": datetime.now().isoformat()
    }

    print(f"    Parameters: {model_info['parameters']:,}")
    print(f"    Layers: {model_info['layers']}")

    # =========================================
    # 2. Validation Metrics
    # =========================================
    print(f"\n[2] Running validation on test set...")

    # Run validation
    results = model.val(
        data=str(data_yaml),
        split='test',
        device=device,
        verbose=True,
        plots=True,
        save_json=True
    )

    # Extract metrics
    metrics = {
        "mAP50": float(results.box.map50),
        "mAP50-95": float(results.box.map),
        "precision": float(results.box.mp),
        "recall": float(results.box.mr),
        "f1_score": 2 * (results.box.mp * results.box.mr) / (results.box.mp + results.box.mr + 1e-10)
    }

    # Per-class metrics
    class_names = model.names
    per_class_metrics = {}

    for i, name in class_names.items():
        if i < len(results.box.ap50):
            per_class_metrics[name] = {
                "AP50": float(results.box.ap50[i]),
                "AP50-95": float(results.box.ap[i]) if i < len(results.box.ap) else 0.0
            }

    print(f"\n" + "=" * 50)
    print("EVALUATION RESULTS")
    print("=" * 50)
    print(f"\nOverall Metrics:")
    print(f"  mAP@50:      {metrics['mAP50']:.4f}")
    print(f"  mAP@50-95:   {metrics['mAP50-95']:.4f}")
    print(f"  Precision:   {metrics['precision']:.4f}")
    print(f"  Recall:      {metrics['recall']:.4f}")
    print(f"  F1-Score:    {metrics['f1_score']:.4f}")

    print(f"\nPer-Class AP@50:")
    for name, m in per_class_metrics.items():
        print(f"  {name:12s}: {m['AP50']:.4f}")

    # =========================================
    # 3. Inference Speed Test
    # =========================================
    print(f"\n[3] Measuring inference speed...")

    import time
    test_img_dir = PROJECT_ROOT / "data" / "xray" / "test" / "images"
    test_images = list(test_img_dir.glob("*.png"))[:20]

    # Warmup
    for img in test_images[:3]:
        _ = model.predict(str(img), verbose=False)

    # Measure
    times = []
    for img in test_images:
        start = time.time()
        _ = model.predict(str(img), verbose=False)
        times.append(time.time() - start)

    speed_metrics = {
        "mean_inference_ms": np.mean(times) * 1000,
        "std_inference_ms": np.std(times) * 1000,
        "fps": 1.0 / np.mean(times),
        "min_ms": np.min(times) * 1000,
        "max_ms": np.max(times) * 1000
    }

    print(f"  Mean inference time: {speed_metrics['mean_inference_ms']:.2f} ms")
    print(f"  FPS: {speed_metrics['fps']:.1f}")

    # =========================================
    # 4. Generate Visualizations
    # =========================================
    print(f"\n[4] Generating visualizations...")

    # 4.1 Training curves (if available)
    run_dir = model_path.parent.parent
    training_df = load_training_results(run_dir)

    if training_df is not None:
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))

        # Loss curves
        ax = axes[0, 0]
        if 'train/box_loss' in training_df.columns:
            ax.plot(training_df['epoch'], training_df['train/box_loss'], label='Box Loss', color='blue')
        if 'train/cls_loss' in training_df.columns:
            ax.plot(training_df['epoch'], training_df['train/cls_loss'], label='Cls Loss', color='red')
        if 'train/dfl_loss' in training_df.columns:
            ax.plot(training_df['epoch'], training_df['train/dfl_loss'], label='DFL Loss', color='green')
        ax.set_xlabel('Epoch')
        ax.set_ylabel('Loss')
        ax.set_title('Training Loss Curves')
        ax.legend()
        ax.grid(True, alpha=0.3)

        # mAP curves
        ax = axes[0, 1]
        if 'metrics/mAP50(B)' in training_df.columns:
            ax.plot(training_df['epoch'], training_df['metrics/mAP50(B)'], label='mAP@50', color='blue')
        if 'metrics/mAP50-95(B)' in training_df.columns:
            ax.plot(training_df['epoch'], training_df['metrics/mAP50-95(B)'], label='mAP@50-95', color='red')
        ax.set_xlabel('Epoch')
        ax.set_ylabel('mAP')
        ax.set_title('mAP Curves')
        ax.legend()
        ax.grid(True, alpha=0.3)

        # Precision/Recall curves
        ax = axes[1, 0]
        if 'metrics/precision(B)' in training_df.columns:
            ax.plot(training_df['epoch'], training_df['metrics/precision(B)'], label='Precision', color='blue')
        if 'metrics/recall(B)' in training_df.columns:
            ax.plot(training_df['epoch'], training_df['metrics/recall(B)'], label='Recall', color='red')
        ax.set_xlabel('Epoch')
        ax.set_ylabel('Value')
        ax.set_title('Precision & Recall Curves')
        ax.legend()
        ax.grid(True, alpha=0.3)

        # Learning rate
        ax = axes[1, 1]
        if 'lr/pg0' in training_df.columns:
            ax.plot(training_df['epoch'], training_df['lr/pg0'], label='LR pg0', color='blue')
        ax.set_xlabel('Epoch')
        ax.set_ylabel('Learning Rate')
        ax.set_title('Learning Rate Schedule')
        ax.legend()
        ax.grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig(output_dir / 'training_curves.png', dpi=150, bbox_inches='tight')
        plt.close()
        print(f"  Saved: training_curves.png")

    # 4.2 Per-class AP bar chart
    fig, ax = plt.subplots(figsize=(10, 6))
    classes = list(per_class_metrics.keys())
    ap50_values = [per_class_metrics[c]['AP50'] for c in classes]

    colors = ['#ff6b6b', '#4ecdc4', '#45b7d1', '#96ceb4', '#ffeaa7', '#74b9ff']
    bars = ax.bar(classes, ap50_values, color=colors[:len(classes)], edgecolor='black', linewidth=1.2)

    ax.set_ylabel('AP@50', fontsize=12)
    ax.set_title('Per-Class Average Precision (AP@50)', fontsize=14, fontweight='bold')
    ax.set_ylim(0, 1.0)
    ax.grid(True, alpha=0.3, axis='y')

    # Add value labels
    for bar, val in zip(bars, ap50_values):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.02,
                f'{val:.3f}', ha='center', va='bottom', fontsize=10)

    plt.tight_layout()
    plt.savefig(output_dir / 'per_class_ap.png', dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved: per_class_ap.png")

    # 4.3 Summary metrics table
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.axis('off')

    table_data = [
        ['Metric', 'Value'],
        ['mAP@50', f'{metrics["mAP50"]:.4f}'],
        ['mAP@50-95', f'{metrics["mAP50-95"]:.4f}'],
        ['Precision', f'{metrics["precision"]:.4f}'],
        ['Recall', f'{metrics["recall"]:.4f}'],
        ['F1-Score', f'{metrics["f1_score"]:.4f}'],
        ['Inference (ms)', f'{speed_metrics["mean_inference_ms"]:.2f}'],
        ['FPS', f'{speed_metrics["fps"]:.1f}']
    ]

    table = ax.table(cellText=table_data[1:], colLabels=table_data[0],
                     loc='center', cellLoc='center',
                     colWidths=[0.4, 0.3])
    table.auto_set_font_size(False)
    table.set_fontsize(12)
    table.scale(1.2, 1.8)

    # Style header
    for j in range(len(table_data[0])):
        table[(0, j)].set_facecolor('#4a90d9')
        table[(0, j)].set_text_props(color='white', fontweight='bold')

    plt.title('Model Performance Summary', fontsize=14, fontweight='bold', pad=20)
    plt.savefig(output_dir / 'metrics_summary.png', dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved: metrics_summary.png")

    # =========================================
    # 5. Save Results to JSON
    # =========================================
    print(f"\n[5] Saving results...")

    full_results = {
        "model_info": model_info,
        "overall_metrics": metrics,
        "per_class_metrics": per_class_metrics,
        "speed_metrics": speed_metrics,
        "dataset_info": {
            "train_images": 350,
            "valid_images": 100,
            "test_images": 50,
            "total_bumps": 98000,
            "classes": list(class_names.values())
        },
        "physics_model": {
            "x_ray_source": "80 kVp Tungsten target",
            "attenuation": "Beer-Lambert Law",
            "noise_model": "Poisson + Gaussian",
            "material": "SAC305 (Lead-free solder)"
        }
    }

    # Save JSON
    json_path = output_dir / 'evaluation_results.json'
    with open(json_path, 'w') as f:
        json.dump(full_results, f, indent=2)
    print(f"  Saved: evaluation_results.json")

    # Save CSV summary
    csv_data = {
        'Metric': ['mAP@50', 'mAP@50-95', 'Precision', 'Recall', 'F1-Score', 'Inference (ms)', 'FPS'],
        'Value': [
            metrics['mAP50'], metrics['mAP50-95'], metrics['precision'],
            metrics['recall'], metrics['f1_score'],
            speed_metrics['mean_inference_ms'], speed_metrics['fps']
        ]
    }
    pd.DataFrame(csv_data).to_csv(output_dir / 'metrics_summary.csv', index=False)
    print(f"  Saved: metrics_summary.csv")

    # Per-class CSV
    class_csv_data = {
        'Class': list(per_class_metrics.keys()),
        'AP@50': [m['AP50'] for m in per_class_metrics.values()],
        'AP@50-95': [m['AP50-95'] for m in per_class_metrics.values()]
    }
    pd.DataFrame(class_csv_data).to_csv(output_dir / 'per_class_metrics.csv', index=False)
    print(f"  Saved: per_class_metrics.csv")

    # =========================================
    # 6. Generate Detection Examples
    # =========================================
    print(f"\n[6] Generating detection examples...")

    example_dir = output_dir / 'detection_examples'
    example_dir.mkdir(exist_ok=True)

    test_images = list(test_img_dir.glob("*.png"))[:5]
    for i, img_path in enumerate(test_images):
        results = model.predict(str(img_path), save=True, project=str(example_dir),
                               name=f'example_{i}', exist_ok=True)
    print(f"  Saved detection examples to: {example_dir}")

    print("\n" + "=" * 70)
    print("EVALUATION COMPLETE")
    print("=" * 70)
    print(f"\nAll results saved to: {output_dir}")
    print(f"\nFiles generated:")
    print(f"  - evaluation_results.json (full results)")
    print(f"  - metrics_summary.csv (overall metrics)")
    print(f"  - per_class_metrics.csv (per-class AP)")
    print(f"  - training_curves.png (loss/mAP curves)")
    print(f"  - per_class_ap.png (AP bar chart)")
    print(f"  - metrics_summary.png (summary table)")
    print(f"  - detection_examples/ (sample detections)")

    return full_results


def main():
    import argparse

    parser = argparse.ArgumentParser(description='YOLOv8 Model Evaluation for Research')
    parser.add_argument('--model', type=str, default=None, help='Path to model weights')
    parser.add_argument('--data', type=str, default=None, help='Path to data.yaml')
    parser.add_argument('--output', type=str, default=None, help='Output directory')

    args = parser.parse_args()

    # Find model
    if args.model:
        model_path = Path(args.model)
    else:
        model_path = find_best_model()

    # Data yaml
    if args.data:
        data_yaml = Path(args.data)
    else:
        data_yaml = PROJECT_ROOT / "data" / "xray" / "data.yaml"

    # Output directory
    if args.output:
        output_dir = Path(args.output)
    else:
        output_dir = PROJECT_ROOT / "evaluation_results" / datetime.now().strftime("%Y%m%d_%H%M%S")

    print(f"Model: {model_path}")
    print(f"Data: {data_yaml}")
    print(f"Output: {output_dir}")

    evaluate_model(model_path, data_yaml, output_dir)


if __name__ == "__main__":
    main()
