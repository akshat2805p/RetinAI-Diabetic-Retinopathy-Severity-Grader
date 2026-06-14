"""
RetinAI — Model Architecture
EfficientNet-B4 NoisyStudent with Generalized Mean (GeM) Pooling head.

Architecture:
    EfficientNet-B4 (feature extractor, 1792-d)
      └─ GeM Pooling (learnable p)
      └─ Dropout(0.4)
      └─ BatchNorm1d(1792)
      └─ Linear(1792 → num_classes)

Usage:
    from src.model import RetinAIModel
    model = RetinAIModel("tf_efficientnet_b4_ns", num_classes=5)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import timm


class GeM(nn.Module):
    """
    Generalized Mean Pooling with learnable exponent p.

    GeM(x) = (1/n · Σ xᵢᵖ)^(1/p)

    When p=1, GeM reduces to Global Average Pooling (GAP).
    When p→∞, GeM approaches Global Max Pooling.
    Learning p allows the model to adaptively weight feature activations.

    Args:
        p:   Initial pooling exponent (default 3)
        eps: Small constant for numerical stability
    """

    def __init__(self, p: float = 3.0, eps: float = 1e-6):
        super().__init__()
        self.p = nn.Parameter(torch.ones(1) * p)
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.adaptive_avg_pool2d(
            x.clamp(min=self.eps).pow(self.p),
            output_size=1
        ).pow(1.0 / self.p)

    def __repr__(self) -> str:
        return f"GeM(p={self.p.data.tolist()[0]:.4f}, eps={self.eps})"


class RetinAIModel(nn.Module):
    """
    EfficientNet-B4 NoisyStudent + GeM Pooling + Dropout Head.

    The backbone is loaded from the `timm` library with ImageNet-21k
    NoisyStudent pretrained weights. The default classifier head and
    global pooling are removed and replaced with our custom GeM head.

    Architecture:
        EfficientNet-B4 (feature extractor, 1792-d output)
          └─ GeM Pooling (1792,1,1) → (1792,)
          └─ Dropout(0.4)
          └─ BatchNorm1d(1792)
          └─ Linear(1792 → num_classes)

    Args:
        model_name:  timm model identifier (e.g., "tf_efficientnet_b4_ns")
        num_classes: Number of output classes (5 for DR grading)
        pretrained:  Whether to load pretrained weights
        dropout:     Dropout rate before the final linear layer
    """

    def __init__(self, model_name: str, num_classes: int,
                 pretrained: bool = True, dropout: float = 0.4):
        super().__init__()
        self.backbone = timm.create_model(
            model_name,
            pretrained=pretrained,
            num_classes=0,          # remove classifier head
            global_pool=""          # disable default pooling
        )
        feat_dim = self.backbone.num_features   # 1792 for B4

        self.pool = GeM(p=3)
        self.drop = nn.Dropout(dropout)
        self.bn = nn.BatchNorm1d(feat_dim)
        self.fc = nn.Linear(feat_dim, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.

        Args:
            x: Input tensor (B, 3, H, W)

        Returns:
            Logits tensor (B, num_classes)
        """
        feats = self.backbone(x)                  # (B, 1792, H, W)
        feats = self.pool(feats).flatten(1)        # (B, 1792)
        feats = self.drop(feats)
        feats = self.bn(feats)
        return self.fc(feats)                      # (B, num_classes)

    def get_cam_layer(self) -> nn.Module:
        """Return the last conv block for GradCAM targeting."""
        return self.backbone.blocks[-1]


if __name__ == "__main__":
    # Quick sanity check
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = RetinAIModel(
        model_name="tf_efficientnet_b4_ns",
        num_classes=5,
        pretrained=False,  # skip download for quick test
    ).to(device)

    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters()
                          if p.requires_grad)

    print(f"Model      : tf_efficientnet_b4_ns")
    print(f"Total Params     : {total_params:,}")
    print(f"Trainable Params : {trainable_params:,}")
    print(f"GeM pooling      : {model.pool}")

    dummy = torch.randn(2, 3, 512, 512).to(device)
    with torch.no_grad():
        out = model(dummy)
    print(f"Forward pass OK  — output shape: {out.shape}")  # (2, 5)
    print("✅ Model architecture OK")
