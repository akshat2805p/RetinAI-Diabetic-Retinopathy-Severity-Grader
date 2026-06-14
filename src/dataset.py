"""
RetinAI — Dataset & Augmentation Pipelines
APTOSDataset class + train/val augmentation transforms.

Usage:
    from src.dataset import APTOSDataset, get_train_transform, get_val_transform
    ds = APTOSDataset(df, img_dir, transform=get_train_transform(512))
"""

from pathlib import Path

import numpy as np
from PIL import Image
import albumentations as A
from albumentations.pytorch import ToTensorV2
import torch
from torch.utils.data import Dataset

from src.preprocess import full_preprocess


# ── ImageNet normalization constants ──────────────────────────────────────────
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]


class APTOSDataset(Dataset):
    """
    PyTorch Dataset for the APTOS 2019 Blindness Detection challenge.

    Each sample is a retinal fundus image (PNG) with an integer DR grade (0-4).
    Images are preprocessed with the Ben Graham + CLAHE pipeline before
    augmentation and normalization.

    Args:
        df:        DataFrame with columns ['id_code', 'diagnosis']
        img_dir:   Path to directory containing image files
        transform: Albumentations Compose pipeline
        is_test:   If True, return label=-1 (for test-time inference)
        img_size:  Target image size for preprocessing (default 512)
    """

    def __init__(self, df, img_dir, transform=None, is_test=False,
                 img_size=512):
        self.df = df.reset_index(drop=True)
        self.img_dir = Path(img_dir)
        self.transform = transform
        self.is_test = is_test
        self.img_size = img_size

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, idx: int):
        row = self.df.iloc[idx]
        img_path = self.img_dir / f"{row['id_code']}.png"

        if img_path.exists():
            img = np.array(Image.open(img_path).convert("RGB"))
            img = full_preprocess(img, self.img_size)
        else:
            # Fallback: synthetic image for portability / testing
            img = np.random.randint(
                0, 255,
                (self.img_size, self.img_size, 3),
                dtype=np.uint8
            )

        if self.transform:
            augmented = self.transform(image=img)
            img = augmented["image"]

        label = int(row["diagnosis"]) if not self.is_test else -1
        return img, label


def get_train_transform(img_size: int = 512) -> A.Compose:
    """
    Training augmentation pipeline.

    Augmentations applied:
        - RandomResizedCrop (scale 0.8-1.0)
        - Horizontal / Vertical Flip
        - RandomRotate90
        - ShiftScaleRotate
        - GaussNoise / GaussianBlur / MotionBlur (OneOf)
        - BrightnessContrast / HueSaturationValue / CLAHE (OneOf)
        - CoarseDropout (Cutout regularization)
        - ImageNet Normalization + ToTensor

    Args:
        img_size: Target image dimension (square)

    Returns:
        Albumentations Compose pipeline
    """
    return A.Compose([
        A.RandomResizedCrop(height=img_size, width=img_size,
                            scale=(0.8, 1.0), p=1.0),
        A.HorizontalFlip(p=0.5),
        A.VerticalFlip(p=0.5),
        A.RandomRotate90(p=0.5),
        A.ShiftScaleRotate(shift_limit=0.05, scale_limit=0.1,
                           rotate_limit=20, p=0.5),
        A.OneOf([
            A.GaussNoise(var_limit=(10, 50)),
            A.GaussianBlur(blur_limit=(3, 5)),
            A.MotionBlur(blur_limit=5),
        ], p=0.3),
        A.OneOf([
            A.RandomBrightnessContrast(brightness_limit=0.2,
                                       contrast_limit=0.2),
            A.HueSaturationValue(hue_shift_limit=10,
                                 sat_shift_limit=20,
                                 val_shift_limit=10),
            A.CLAHE(clip_limit=4.0, p=1.0),
        ], p=0.4),
        A.CoarseDropout(max_holes=8, max_height=img_size // 20,
                        max_width=img_size // 20, p=0.2),
        A.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
        ToTensorV2(),
    ])


def get_val_transform(img_size: int = 512) -> A.Compose:
    """
    Validation / inference transform (no augmentation).

    Only resizes, normalizes, and converts to tensor.

    Args:
        img_size: Target image dimension (square)

    Returns:
        Albumentations Compose pipeline
    """
    return A.Compose([
        A.Resize(img_size, img_size),
        A.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
        ToTensorV2(),
    ])


if __name__ == "__main__":
    import pandas as pd

    # Create a small synthetic dataset for testing
    n = 20
    np.random.seed(42)
    df = pd.DataFrame({
        "id_code": [f"img_{i:05d}" for i in range(n)],
        "diagnosis": np.random.choice([0, 1, 2, 3, 4], size=n,
                                       p=[0.49, 0.10, 0.27, 0.06, 0.08])
    })

    transform = get_train_transform(512)
    ds = APTOSDataset(df, "./data/aptos2019/train_images",
                      transform=transform)

    img, label = ds[0]
    print(f"Dataset length : {len(ds)}")
    print(f"Image shape    : {img.shape}")
    print(f"Image dtype    : {img.dtype}")
    print(f"Label          : {label}")
    print("✅ APTOSDataset OK")
