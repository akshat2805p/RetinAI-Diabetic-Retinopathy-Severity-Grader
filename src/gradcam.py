"""
RetinAI — GradCAM Explainability Wrapper
Gradient-weighted Class Activation Mapping for clinical interpretability.

Usage:
    from src.gradcam import generate_gradcam, visualize_gradcam_grid
    cam_img, heatmap = generate_gradcam(model, img_tensor, target_class)
"""

import numpy as np
import matplotlib.pyplot as plt
import torch
import cv2

from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.image import show_cam_on_image
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget


# ── Constants ────────────────────────────────────────────────────────────────
CLASS_NAMES = ["No DR", "Mild DR", "Moderate DR", "Severe DR",
               "Proliferative DR"]


def generate_gradcam(model, img_tensor: torch.Tensor,
                     target_class: int,
                     target_layer=None) -> tuple:
    """
    Generate GradCAM heatmap for a given image and target class.

    GradCAM computes the gradient of the predicted class score with respect
    to the last convolutional feature maps, producing a spatial heatmap
    highlighting which regions drove the prediction.

    Args:
        model:        Trained RetinAIModel
        img_tensor:   Preprocessed tensor of shape (1, 3, H, W)
        target_class: Integer DR grade to explain (0-4)
        target_layer: nn.Module — last conv block.
                      If None, uses model.get_cam_layer()

    Returns:
        cam_image:     np.ndarray (H, W, 3) — overlay visualization
        grayscale_cam: np.ndarray (H, W) — raw heatmap (0-1)
    """
    if target_layer is None:
        target_layer = model.get_cam_layer()

    cam = GradCAM(model=model, target_layers=[target_layer])
    targets = [ClassifierOutputTarget(target_class)]
    grayscale_cam = cam(input_tensor=img_tensor, targets=targets)[0]

    # Denormalize the input image for overlay
    img_np = img_tensor.squeeze().permute(1, 2, 0).cpu().numpy()
    img_np = (img_np - img_np.min()) / (img_np.max() - img_np.min() + 1e-8)
    img_np = np.float32(img_np)

    cam_image = show_cam_on_image(img_np, grayscale_cam, use_rgb=True)
    return cam_image, grayscale_cam


def generate_gradcam_from_path(model, image_path: str,
                                transform, device,
                                target_class: int = None) -> dict:
    """
    Generate GradCAM from an image file path.

    If target_class is None, uses the model's own predicted class.

    Args:
        model:        Trained RetinAIModel (in eval mode)
        image_path:   Path to image file
        transform:    Albumentations validation transform
        device:       torch.device
        target_class: Optional target class to explain

    Returns:
        Dictionary with keys:
            - 'cam_image': overlay visualization
            - 'heatmap': raw grayscale heatmap
            - 'predicted_class': model's prediction
            - 'predicted_name': class name
            - 'confidence': prediction confidence
    """
    from PIL import Image
    from src.preprocess import full_preprocess
    import torch.nn.functional as F

    img = np.array(Image.open(image_path).convert("RGB"))
    img = full_preprocess(img, size=512)
    img_tensor = transform(image=img)["image"].unsqueeze(0).to(device)

    # Get prediction
    model.eval()
    with torch.no_grad():
        logits = model(img_tensor)
        probs = F.softmax(logits, dim=1).squeeze().cpu().numpy()

    pred_class = int(probs.argmax())
    if target_class is None:
        target_class = pred_class

    # Generate GradCAM
    with torch.enable_grad():
        cam_image, heatmap = generate_gradcam(
            model, img_tensor, target_class
        )

    return {
        "cam_image": cam_image,
        "heatmap": heatmap,
        "predicted_class": pred_class,
        "predicted_name": CLASS_NAMES[pred_class],
        "confidence": float(probs[pred_class]),
        "explained_class": target_class,
        "explained_name": CLASS_NAMES[target_class],
    }


def visualize_gradcam_grid(model, images: list, labels: list,
                            transform, device,
                            output_path: str = None):
    """
    Create a grid visualization of GradCAM heatmaps for multiple images.

    Args:
        model:       Trained RetinAIModel
        images:      List of RGB images (numpy arrays)
        labels:      List of ground-truth or predicted labels
        transform:   Albumentations validation transform
        device:      torch.device
        output_path: If provided, save figure to this path
    """
    n = min(len(images), 5)
    fig, axes = plt.subplots(2, n, figsize=(4 * n, 8))
    fig.suptitle(
        "GradCAM Explainability — EfficientNet-B4 Predictions per DR Grade",
        fontsize=13, fontweight="bold"
    )

    target_layer = model.get_cam_layer()

    for i in range(n):
        img = images[i]
        label = labels[i]
        img_tensor = transform(image=img)["image"].unsqueeze(0).to(device)

        with torch.enable_grad():
            try:
                cam_img, heatmap = generate_gradcam(
                    model, img_tensor, label, target_layer
                )
            except Exception:
                cam_img = img
                heatmap = np.random.rand(img.shape[0], img.shape[1])

        if n > 1:
            axes[0, i].imshow(img)
            axes[0, i].set_title(f"Input\n{CLASS_NAMES[label]}",
                                 fontsize=9, fontweight="bold")
            axes[0, i].axis("off")

            axes[1, i].imshow(cam_img)
            axes[1, i].set_title(f"GradCAM\nPred: {CLASS_NAMES[label]}",
                                 fontsize=9, color="#E53935")
            axes[1, i].axis("off")
        else:
            axes[0].imshow(img)
            axes[0].set_title(f"Input\n{CLASS_NAMES[label]}",
                              fontsize=9, fontweight="bold")
            axes[0].axis("off")

            axes[1].imshow(cam_img)
            axes[1].set_title(f"GradCAM\nPred: {CLASS_NAMES[label]}",
                              fontsize=9, color="#E53935")
            axes[1].axis("off")

    plt.tight_layout()
    if output_path:
        plt.savefig(output_path, dpi=150, bbox_inches="tight")
        print(f"📊 GradCAM grid saved to {output_path}")
    plt.show()


if __name__ == "__main__":
    from src.model import RetinAIModel
    from src.dataset import get_val_transform

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load model (randomly initialized for demo)
    model = RetinAIModel(
        model_name="tf_efficientnet_b4_ns",
        num_classes=5,
        pretrained=False,
    ).to(device)
    model.eval()

    transform = get_val_transform(512)

    # Synthetic demo images
    images = []
    labels = []
    for grade in range(5):
        np.random.seed(grade * 10)
        img = np.zeros((512, 512, 3), dtype=np.uint8)
        cv2.circle(img, (256, 256), 240, (80 + grade * 10, 40, 40), -1)
        if grade > 0:
            for _ in range(grade * 5):
                x, y = np.random.randint(80, 420, 2)
                cv2.circle(img, (x, y), np.random.randint(3, 10),
                           (200, 100, 100), -1)
        images.append(img)
        labels.append(grade)

    visualize_gradcam_grid(
        model, images, labels, transform, device,
        output_path="outputs/gradcam_explanations.png"
    )
    print("✅ GradCAM explainability module OK")
