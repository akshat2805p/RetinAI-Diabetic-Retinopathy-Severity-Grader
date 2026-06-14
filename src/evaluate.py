"""
RetinAI — Evaluation Module
Metrics computation, confusion matrix, and ROC curve visualization.

Usage:
    python src/evaluate.py --model_path models/best_fold0.pth --data_dir ./data/aptos2019

    Or import:
    from src.evaluate import compute_metrics, plot_confusion_matrix, plot_roc_curves
"""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.cuda.amp import autocast
from sklearn.metrics import (
    accuracy_score, f1_score, cohen_kappa_score,
    confusion_matrix, classification_report,
    roc_curve, auc
)
from sklearn.preprocessing import label_binarize
from tqdm import tqdm

from src.model import RetinAIModel
from src.dataset import APTOSDataset, get_val_transform


# ── Constants ────────────────────────────────────────────────────────────────
CLASS_NAMES = ["No DR", "Mild DR", "Moderate DR", "Severe DR",
               "Proliferative DR"]
COLORS = ["#4CAF50", "#FFC107", "#FF9800", "#F44336", "#9C27B0"]


def compute_metrics(y_true: np.ndarray,
                    y_pred: np.ndarray,
                    y_probs: np.ndarray = None) -> dict:
    """
    Compute all evaluation metrics.

    Args:
        y_true:  Ground truth labels (n,)
        y_pred:  Predicted labels (n,)
        y_probs: Predicted probabilities (n, 5) — optional for AUC

    Returns:
        Dictionary of metric name → value
    """
    metrics = {
        "accuracy": accuracy_score(y_true, y_pred),
        "qwk": cohen_kappa_score(y_true, y_pred, weights="quadratic"),
        "macro_f1": f1_score(y_true, y_pred, average="macro",
                             zero_division=0),
        "weighted_f1": f1_score(y_true, y_pred, average="weighted",
                                zero_division=0),
    }

    if y_probs is not None:
        y_bin = label_binarize(y_true, classes=[0, 1, 2, 3, 4])
        per_class_auc = []
        for i in range(5):
            try:
                fpr, tpr, _ = roc_curve(y_bin[:, i], y_probs[:, i])
                per_class_auc.append(auc(fpr, tpr))
            except ValueError:
                per_class_auc.append(0.0)
        metrics["mean_auc"] = np.mean(per_class_auc)
        metrics["per_class_auc"] = dict(zip(CLASS_NAMES, per_class_auc))

    return metrics


def plot_confusion_matrix(y_true: np.ndarray, y_pred: np.ndarray,
                          output_path: str = None):
    """
    Plot raw and normalized confusion matrices side by side.

    Args:
        y_true:      Ground truth labels
        y_pred:      Predicted labels
        output_path: If provided, save figure to this path
    """
    cm = confusion_matrix(y_true, y_pred)
    cm_norm = cm.astype(float) / cm.sum(axis=1, keepdims=True)

    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    fig.suptitle("Model Evaluation — EfficientNet-B4 NoisyStudent",
                 fontsize=13, fontweight="bold")

    # Raw counts
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=CLASS_NAMES, yticklabels=CLASS_NAMES,
                linewidths=0.5, ax=axes[0])
    axes[0].set_title("Confusion Matrix (counts)", fontsize=11)
    axes[0].set_xlabel("Predicted")
    axes[0].set_ylabel("True")
    axes[0].tick_params(axis="x", rotation=30)
    axes[0].tick_params(axis="y", rotation=0)

    # Normalized
    sns.heatmap(cm_norm, annot=True, fmt=".2f", cmap="YlOrRd",
                xticklabels=CLASS_NAMES, yticklabels=CLASS_NAMES,
                linewidths=0.5, vmin=0, vmax=1, ax=axes[1])
    axes[1].set_title("Confusion Matrix (normalized)", fontsize=11)
    axes[1].set_xlabel("Predicted")
    axes[1].set_ylabel("True")
    axes[1].tick_params(axis="x", rotation=30)
    axes[1].tick_params(axis="y", rotation=0)

    plt.tight_layout()
    if output_path:
        plt.savefig(output_path, dpi=150, bbox_inches="tight")
        print(f"📊 Confusion matrix saved to {output_path}")
    plt.show()


def plot_roc_curves(y_true: np.ndarray, y_probs: np.ndarray,
                    output_path: str = None):
    """
    Plot One-vs-Rest ROC curves for all 5 DR classes.

    Args:
        y_true:      Ground truth labels
        y_probs:     Predicted probabilities (n, 5)
        output_path: If provided, save figure to this path
    """
    y_bin = label_binarize(y_true, classes=[0, 1, 2, 3, 4])

    fig, ax = plt.subplots(figsize=(9, 7))
    aucs = []

    for i, (cls, color) in enumerate(zip(CLASS_NAMES, COLORS)):
        fpr, tpr, _ = roc_curve(y_bin[:, i], y_probs[:, i])
        roc_auc = auc(fpr, tpr)
        aucs.append(roc_auc)
        ax.plot(fpr, tpr, color=color, lw=2,
                label=f"{cls} (AUC = {roc_auc:.3f})")

    ax.plot([0, 1], [0, 1], "k--", lw=1, alpha=0.5,
            label="Random (AUC = 0.500)")
    ax.fill_between([0, 1], [0, 1], [0, 1], alpha=0.03, color="gray")
    ax.set_xlim([-0.01, 1.01])
    ax.set_ylim([-0.01, 1.02])
    ax.set_xlabel("False Positive Rate", fontsize=12)
    ax.set_ylabel("True Positive Rate", fontsize=12)
    ax.set_title(
        f"ROC Curves — One-vs-Rest | Mean AUC = {np.mean(aucs):.3f}",
        fontsize=13, fontweight="bold"
    )
    ax.legend(loc="lower right", fontsize=10)
    ax.grid(alpha=0.3)

    plt.tight_layout()
    if output_path:
        plt.savefig(output_path, dpi=150, bbox_inches="tight")
        print(f"📊 ROC curves saved to {output_path}")
    plt.show()

    return aucs


@torch.no_grad()
def evaluate_model(model, loader, device, use_amp=True):
    """
    Run inference on the validation set and collect predictions.

    Args:
        model:   RetinAIModel (in eval mode)
        loader:  Validation DataLoader
        device:  torch.device
        use_amp: Whether to use mixed precision

    Returns:
        (probs, labels, preds): Arrays of shape (n, 5), (n,), (n,)
    """
    model.eval()
    preds_all, labels_all, probs_all = [], [], []

    for imgs, labels in tqdm(loader, desc="Evaluating", leave=False):
        imgs = imgs.to(device)
        with autocast(enabled=use_amp):
            logits = model(imgs)

        probs = F.softmax(logits, dim=1)
        preds_all.extend(logits.argmax(1).cpu().numpy())
        labels_all.extend(labels.numpy())
        probs_all.extend(probs.cpu().numpy())

    return (np.array(probs_all), np.array(labels_all),
            np.array(preds_all))


def main(args):
    """Run full evaluation pipeline."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # ── Load model ───────────────────────────────────────────────────────
    model = RetinAIModel(
        model_name="tf_efficientnet_b4_ns",
        num_classes=5,
        pretrained=False,
    ).to(device)

    model_path = Path(args.model_path)
    if model_path.exists():
        model.load_state_dict(torch.load(model_path, map_location=device))
        print(f"✅ Loaded model from {model_path}")
    else:
        print(f"⚠️  Model checkpoint not found at {model_path}")
        print("   Using randomly initialized model for demo")

    model.eval()

    # ── Load data ────────────────────────────────────────────────────────
    data_dir = Path(args.data_dir)
    csv_path = data_dir / "train.csv"

    if csv_path.exists():
        df = pd.read_csv(csv_path)
    else:
        print("⚠️  Using synthetic data for demo")
        np.random.seed(42)
        n = 732
        true_labels = np.random.choice(
            [0, 1, 2, 3, 4], size=n,
            p=[0.490, 0.096, 0.270, 0.064, 0.080]
        )
        df = pd.DataFrame({
            "id_code": [f"img_{i:05d}" for i in range(n)],
            "diagnosis": true_labels
        })

    transform = get_val_transform(args.img_size)
    ds = APTOSDataset(df, data_dir / "train_images", transform=transform,
                      img_size=args.img_size)
    loader = DataLoader(ds, batch_size=args.batch_size * 2, shuffle=False,
                        num_workers=args.num_workers)

    # ── Evaluate ─────────────────────────────────────────────────────────
    probs, labels, preds = evaluate_model(model, loader, device)

    # ── Metrics ──────────────────────────────────────────────────────────
    metrics = compute_metrics(labels, preds, probs)

    print("\n" + "=" * 50)
    print("   FINAL EVALUATION METRICS")
    print("=" * 50)
    print(f"  Accuracy                : {metrics['accuracy']:.4f}")
    print(f"  Quadratic Weighted Kappa: {metrics['qwk']:.4f}")
    print(f"  Macro F1-Score          : {metrics['macro_f1']:.4f}")
    print(f"  Weighted F1-Score       : {metrics['weighted_f1']:.4f}")
    if "mean_auc" in metrics:
        print(f"  Mean AUC-ROC            : {metrics['mean_auc']:.4f}")
    print("=" * 50)
    print()
    print(classification_report(labels, preds,
                                 target_names=CLASS_NAMES, zero_division=0))

    # ── Plots ────────────────────────────────────────────────────────────
    plot_confusion_matrix(
        labels, preds,
        output_path=str(output_dir / "confusion_matrix.png")
    )
    plot_roc_curves(
        labels, probs,
        output_path=str(output_dir / "roc_curves.png")
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="RetinAI — Evaluate DR Grading Model"
    )
    parser.add_argument("--model_path", type=str,
                        default="models/best_fold0.pth",
                        help="Path to model checkpoint")
    parser.add_argument("--data_dir", type=str, default="./data/aptos2019",
                        help="Path to dataset directory")
    parser.add_argument("--output_dir", type=str, default="./outputs",
                        help="Path to save evaluation outputs")
    parser.add_argument("--img_size", type=int, default=512,
                        help="Input image size")
    parser.add_argument("--batch_size", type=int, default=16,
                        help="Batch size")
    parser.add_argument("--num_workers", type=int, default=4,
                        help="DataLoader workers")

    args = parser.parse_args()
    main(args)
