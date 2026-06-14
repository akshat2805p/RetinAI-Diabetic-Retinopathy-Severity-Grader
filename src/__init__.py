"""
RetinAI — Source Package

Core modules for the Diabetic Retinopathy grading pipeline:

    src.preprocess  — Ben Graham + CLAHE image preprocessing
    src.model       — EfficientNet-B4 + GeM pooling architecture
    src.dataset     — APTOSDataset + augmentation pipelines
    src.train       — Training loop with AMP + early stopping
    src.evaluate    — Metrics, confusion matrix, ROC curves
    src.gradcam     — GradCAM explainability wrapper
"""

from src.preprocess import full_preprocess
from src.model import RetinAIModel, GeM
from src.dataset import APTOSDataset, get_train_transform, get_val_transform

__all__ = [
    "full_preprocess",
    "RetinAIModel",
    "GeM",
    "APTOSDataset",
    "get_train_transform",
    "get_val_transform",
]
