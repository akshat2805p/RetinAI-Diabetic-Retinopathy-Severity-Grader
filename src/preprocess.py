"""
RetinAI — Preprocessing Pipeline
Ben Graham retinal enhancement + CLAHE contrast normalization + circular crop.

Usage:
    from src.preprocess import full_preprocess
    processed = full_preprocess(img_rgb, size=512)
"""

import cv2
import numpy as np


def ben_graham_preprocess(img: np.ndarray, sigmaX: int = 10) -> np.ndarray:
    """
    Ben Graham retinal image enhancement.
    Removes illumination variation via Gaussian subtraction — winner of the 2015
    Kaggle Diabetic Retinopathy competition.

    Steps:
        1. Multiply image by 4
        2. Subtract Gaussian-blurred version (×4)
        3. Add 128 bias to center the intensities

    Args:
        img:    RGB image as uint8 ndarray (H, W, 3)
        sigmaX: Gaussian kernel sigma (higher → stronger smoothing)

    Returns:
        Enhanced RGB image (uint8)
    """
    img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    img = cv2.addWeighted(img, 4,
                          cv2.GaussianBlur(img, (0, 0), sigmaX), -4, 128)
    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)


def clahe_preprocess(img: np.ndarray, clip_limit: float = 2.0,
                     tile_grid: tuple = (8, 8)) -> np.ndarray:
    """
    Apply Contrast Limited Adaptive Histogram Equalization (CLAHE) on the
    L-channel of the LAB color space.

    CLAHE enhances local contrast without amplifying noise by limiting the
    histogram equalization to small tiles and clipping extreme peaks.

    Args:
        img:        RGB image as uint8 ndarray (H, W, 3)
        clip_limit: CLAHE clip limit (higher → more contrast)
        tile_grid:  Tile grid size for local equalization

    Returns:
        Enhanced RGB image (uint8)
    """
    lab = cv2.cvtColor(img, cv2.COLOR_RGB2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=tile_grid)
    l = clahe.apply(l)
    lab = cv2.merge([l, a, b])
    return cv2.cvtColor(lab, cv2.COLOR_LAB2RGB)


def crop_black_borders(img: np.ndarray, tol: int = 7) -> np.ndarray:
    """
    Remove dark borders from retinal fundus images.

    Many fundus cameras capture circular images on a dark background.
    This function finds the bounding box of non-dark pixels and crops to it.

    Args:
        img: RGB or grayscale image (ndarray)
        tol: Intensity threshold — pixels below this are considered "dark"

    Returns:
        Cropped image (ndarray)
    """
    mask = img > tol
    if img.ndim == 3:
        mask = mask.any(2)
    rows, cols = np.where(mask)
    if rows.size == 0:
        return img
    return img[rows.min():rows.max() + 1, cols.min():cols.max() + 1]


def full_preprocess(img: np.ndarray, size: int = 512) -> np.ndarray:
    """
    Complete preprocessing pipeline combining all stages:
        1. Circular border crop  → remove black background
        2. Resize to target      → standardize dimensions
        3. CLAHE                  → enhance local contrast
        4. Ben Graham             → remove illumination bias

    Args:
        img:  RGB image as uint8 ndarray (H, W, 3)
        size: Target square dimension (default 512×512)

    Returns:
        Preprocessed RGB image (uint8, size × size × 3)
    """
    img = crop_black_borders(img)
    img = cv2.resize(img, (size, size))
    img = clahe_preprocess(img)
    img = ben_graham_preprocess(img)
    return img


if __name__ == "__main__":
    # Quick sanity check — create synthetic image, preprocess, and print shape
    demo = np.zeros((400, 400, 3), dtype=np.uint8)
    cv2.circle(demo, (200, 200), 190, (80, 50, 50), -1)
    cv2.circle(demo, (200, 200), 30, (255, 220, 180), -1)
    demo_rgb = cv2.cvtColor(demo, cv2.COLOR_BGR2RGB)

    result = full_preprocess(demo_rgb, size=512)
    print(f"Input shape:  {demo_rgb.shape}")
    print(f"Output shape: {result.shape}")
    print(f"Output dtype: {result.dtype}")
    print("✅ Preprocessing pipeline OK")
