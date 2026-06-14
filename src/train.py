"""
RetinAI — Training Loop
Mixed precision training with AMP, cosine annealing, early stopping,
and label-smoothed cross-entropy with class weights.

Usage:
    python src/train.py --fold 0 --epochs 30 --img_size 512 --batch_size 16
"""

import os
import gc
import argparse
import random
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.cuda.amp import GradScaler, autocast
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score, f1_score, cohen_kappa_score
from tqdm import tqdm

from src.model import RetinAIModel
from src.dataset import APTOSDataset, get_train_transform, get_val_transform

warnings.filterwarnings("ignore")


# ── Reproducibility ──────────────────────────────────────────────────────────
SEED = 42


def seed_everything(seed: int = SEED):
    """Set seeds for reproducibility across all libraries."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    torch.backends.cudnn.deterministic = True


# ── Loss function ────────────────────────────────────────────────────────────
class LabelSmoothingCrossEntropy(nn.Module):
    """
    Cross-entropy with label smoothing for overconfidence regularization.

    Instead of using hard one-hot targets, distributes ε probability mass
    uniformly across non-target classes. This prevents the model from
    becoming overconfident and improves generalization.

    Args:
        smoothing: Label smoothing factor ε (default 0.05)
        weight:    Per-class weights for handling class imbalance
    """

    def __init__(self, smoothing: float = 0.05, weight=None):
        super().__init__()
        self.smoothing = smoothing
        self.weight = weight

    def forward(self, pred: torch.Tensor,
                target: torch.Tensor) -> torch.Tensor:
        n_classes = pred.size(1)
        log_prob = F.log_softmax(pred, dim=1)

        with torch.no_grad():
            smooth_target = torch.full_like(
                log_prob, self.smoothing / (n_classes - 1)
            )
            smooth_target.scatter_(1, target.unsqueeze(1),
                                   1.0 - self.smoothing)

        if self.weight is not None:
            w = self.weight.to(pred.device)[target]
            loss = -(smooth_target * log_prob).sum(dim=1)
            return (loss * w).mean()

        return -(smooth_target * log_prob).sum(dim=1).mean()


# ── Metrics ──────────────────────────────────────────────────────────────────
def quadratic_weighted_kappa(y_true, y_pred) -> float:
    """
    Cohen's Kappa with quadratic weighting — official APTOS competition metric.

    QWK penalizes predictions more heavily when they are further from the
    true label (e.g., predicting 0 when truth is 4 is penalized more than
    predicting 1 when truth is 0).
    """
    return cohen_kappa_score(y_true, y_pred, weights="quadratic")


# ── Training functions ───────────────────────────────────────────────────────
def train_one_epoch(model, loader, criterion, optimizer, scaler, epoch,
                    device, use_amp=True):
    """
    Train for one epoch with mixed precision.

    Args:
        model:     RetinAIModel
        loader:    Training DataLoader
        criterion: Loss function
        optimizer: AdamW optimizer
        scaler:    GradScaler for mixed precision
        epoch:     Current epoch number
        device:    torch.device
        use_amp:   Whether to use automatic mixed precision

    Returns:
        (avg_loss, qwk): Training loss and Quadratic Weighted Kappa
    """
    model.train()
    total_loss = 0.0
    preds_all, labels_all = [], []

    pbar = tqdm(loader, desc=f"Epoch {epoch:02d} [Train]", leave=False)
    for imgs, labels in pbar:
        imgs, labels = imgs.to(device), labels.to(device)
        optimizer.zero_grad()

        with autocast(enabled=use_amp):
            logits = model(imgs)
            loss = criterion(logits, labels)

        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        scaler.step(optimizer)
        scaler.update()

        total_loss += loss.item() * imgs.size(0)
        preds_all.extend(logits.argmax(1).cpu().numpy())
        labels_all.extend(labels.cpu().numpy())
        pbar.set_postfix(loss=f"{loss.item():.4f}")

    avg_loss = total_loss / len(loader.dataset)
    qwk = quadratic_weighted_kappa(labels_all, preds_all)
    return avg_loss, qwk


@torch.no_grad()
def validate(model, loader, criterion, device, use_amp=True):
    """
    Validate the model on the validation set.

    Args:
        model:     RetinAIModel
        loader:    Validation DataLoader
        criterion: Loss function
        device:    torch.device
        use_amp:   Whether to use automatic mixed precision

    Returns:
        (avg_loss, qwk, accuracy, f1, probs, labels, preds)
    """
    model.eval()
    total_loss = 0.0
    preds_all, labels_all, probs_all = [], [], []

    for imgs, labels in tqdm(loader, desc="Validating", leave=False):
        imgs, labels = imgs.to(device), labels.to(device)
        with autocast(enabled=use_amp):
            logits = model(imgs)
            loss = criterion(logits, labels)

        probs = F.softmax(logits, dim=1)
        total_loss += loss.item() * imgs.size(0)
        preds_all.extend(logits.argmax(1).cpu().numpy())
        labels_all.extend(labels.cpu().numpy())
        probs_all.extend(probs.cpu().numpy())

    avg_loss = total_loss / len(loader.dataset)
    qwk = quadratic_weighted_kappa(labels_all, preds_all)
    acc = accuracy_score(labels_all, preds_all)
    f1 = f1_score(labels_all, preds_all, average="macro", zero_division=0)
    return (avg_loss, qwk, acc, f1,
            np.array(probs_all), np.array(labels_all), np.array(preds_all))


# ── Main training script ────────────────────────────────────────────────────
def main(args):
    """Run the full training pipeline."""
    seed_everything(SEED)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    output_dir = Path(args.output_dir)
    model_dir = Path(args.model_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    model_dir.mkdir(parents=True, exist_ok=True)

    # ── Load data ────────────────────────────────────────────────────────
    data_dir = Path(args.data_dir)
    csv_path = data_dir / "train.csv"

    if csv_path.exists():
        df = pd.read_csv(csv_path)
        print(f"Loaded dataset: {len(df)} samples")
    else:
        # Synthetic data for demo/testing
        print("⚠️  Dataset CSV not found — using synthetic data for demo")
        np.random.seed(42)
        n = 3662
        labels = np.random.choice(
            [0, 1, 2, 3, 4], size=n,
            p=[0.490, 0.096, 0.270, 0.064, 0.080]
        )
        df = pd.DataFrame({
            "id_code": [f"img_{i:05d}" for i in range(n)],
            "diagnosis": labels
        })

    # ── Stratified K-Fold ────────────────────────────────────────────────
    skf = StratifiedKFold(n_splits=args.n_folds, shuffle=True,
                          random_state=SEED)

    for fold, (train_idx, val_idx) in enumerate(
            skf.split(df, df["diagnosis"])):
        if fold != args.fold:
            continue

        print(f"\n{'='*60}")
        print(f"  FOLD {fold} — Training")
        print(f"{'='*60}")

        train_df = df.iloc[train_idx]
        val_df = df.iloc[val_idx]
        print(f"  Train: {len(train_df)} | Val: {len(val_df)}")

        # ── Datasets & Loaders ───────────────────────────────────────────
        img_dir = data_dir / "train_images"
        train_ds = APTOSDataset(train_df, img_dir,
                                transform=get_train_transform(args.img_size),
                                img_size=args.img_size)
        val_ds = APTOSDataset(val_df, img_dir,
                              transform=get_val_transform(args.img_size),
                              img_size=args.img_size)

        train_loader = DataLoader(
            train_ds, batch_size=args.batch_size,
            shuffle=True, num_workers=args.num_workers,
            pin_memory=True, drop_last=True
        )
        val_loader = DataLoader(
            val_ds, batch_size=args.batch_size * 2,
            shuffle=False, num_workers=args.num_workers,
            pin_memory=True
        )

        # ── Model ────────────────────────────────────────────────────────
        model = RetinAIModel(
            model_name=args.model_name,
            num_classes=5,
            pretrained=True,
        ).to(device)

        # ── Class-weighted loss ──────────────────────────────────────────
        class_counts = np.bincount(train_df["diagnosis"], minlength=5)
        class_weights = torch.tensor(
            1.0 / (class_counts + 1e-6), dtype=torch.float32
        )
        class_weights = class_weights / class_weights.sum() * 5  # normalize

        criterion = LabelSmoothingCrossEntropy(
            smoothing=args.label_smooth,
            weight=class_weights
        )

        # ── Optimizer & Scheduler ────────────────────────────────────────
        optimizer = optim.AdamW(
            model.parameters(),
            lr=args.lr,
            weight_decay=args.weight_decay
        )
        scheduler = optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=args.epochs, eta_min=1e-7
        )
        scaler = GradScaler(enabled=args.use_amp)

        # ── Training loop ────────────────────────────────────────────────
        best_qwk = 0.0
        patience_counter = 0
        history = {
            "train_loss": [], "val_loss": [],
            "train_qwk": [], "val_qwk": [],
            "val_acc": [], "val_f1": []
        }

        for epoch in range(1, args.epochs + 1):
            t_loss, t_qwk = train_one_epoch(
                model, train_loader, criterion, optimizer, scaler,
                epoch, device, args.use_amp
            )
            v_loss, v_qwk, v_acc, v_f1, _, _, _ = validate(
                model, val_loader, criterion, device, args.use_amp
            )
            scheduler.step()

            # Record history
            history["train_loss"].append(t_loss)
            history["val_loss"].append(v_loss)
            history["train_qwk"].append(t_qwk)
            history["val_qwk"].append(v_qwk)
            history["val_acc"].append(v_acc)
            history["val_f1"].append(v_f1)

            # Early stopping on validation QWK
            if v_qwk > best_qwk:
                best_qwk = v_qwk
                patience_counter = 0
                save_path = model_dir / f"best_fold{fold}.pth"
                torch.save(model.state_dict(), save_path)
                print(f"  💾 Saved best model (QWK={best_qwk:.4f})")
            else:
                patience_counter += 1
                if patience_counter >= args.patience:
                    print(f"  ⏹ Early stopping at epoch {epoch}")
                    break

            lr_now = optimizer.param_groups[0]["lr"]
            print(
                f"  E{epoch:02d} | "
                f"T_loss={t_loss:.4f} T_qwk={t_qwk:.4f} | "
                f"V_loss={v_loss:.4f} V_qwk={v_qwk:.4f} "
                f"V_acc={v_acc:.4f} V_f1={v_f1:.4f} | "
                f"LR={lr_now:.2e}"
            )

        print(f"\n✅ Training complete — Best QWK: {best_qwk:.4f}")

        # ── Save history ─────────────────────────────────────────────────
        hist_df = pd.DataFrame(history)
        hist_df.to_csv(output_dir / f"training_history_fold{fold}.csv",
                       index=False)
        print(f"📊 History saved to {output_dir}/training_history_fold{fold}.csv")

    # Cleanup
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="RetinAI — Train DR Grading Model"
    )
    parser.add_argument("--data_dir", type=str, default="./data/aptos2019",
                        help="Path to dataset directory")
    parser.add_argument("--output_dir", type=str, default="./outputs",
                        help="Path to save outputs")
    parser.add_argument("--model_dir", type=str, default="./models",
                        help="Path to save model checkpoints")
    parser.add_argument("--model_name", type=str,
                        default="tf_efficientnet_b4_ns",
                        help="timm model name")
    parser.add_argument("--fold", type=int, default=0,
                        help="Which CV fold to train (0-4)")
    parser.add_argument("--n_folds", type=int, default=5,
                        help="Number of CV folds")
    parser.add_argument("--epochs", type=int, default=30,
                        help="Maximum training epochs")
    parser.add_argument("--img_size", type=int, default=512,
                        help="Input image size")
    parser.add_argument("--batch_size", type=int, default=16,
                        help="Batch size")
    parser.add_argument("--num_workers", type=int, default=4,
                        help="DataLoader workers")
    parser.add_argument("--lr", type=float, default=1e-4,
                        help="Learning rate")
    parser.add_argument("--weight_decay", type=float, default=1e-5,
                        help="Weight decay")
    parser.add_argument("--label_smooth", type=float, default=0.05,
                        help="Label smoothing epsilon")
    parser.add_argument("--patience", type=int, default=7,
                        help="Early stopping patience")
    parser.add_argument("--use_amp", action="store_true", default=True,
                        help="Use mixed precision training")

    args = parser.parse_args()
    main(args)
