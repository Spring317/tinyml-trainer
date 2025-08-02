import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import (
    ConvNeXt_Tiny_Weights,
    convnext_large,
    ConvNeXt_Large_Weights,
    convnext_tiny,
    convnext_base,
    ConvNeXt_Base_Weights,
)


class ConvNeXt160(nn.Module):
    def __init__(
        self, num_classes: int, pretrained: bool = True, convext_ver: str = "large"
    ):
        """
        ConvNeXt wrapper for 160x160 input images.

        Args:
            num_classes (int): Number of output classes.
            pretrained (bool): Whether to load ImageNet pretrained weights.
        """
        super(ConvNeXt160, self).__init__()

        if pretrained and convext_ver == "large":
            weights = ConvNeXt_Large_Weights.IMAGENET1K_V1
            self.model = convnext_large(weights=weights)
            self.mean = weights.meta["mean"]
            self.std = weights.meta["std"]
        elif pretrained and convext_ver == "base":
            weights = ConvNeXt_Base_Weights.IMAGENET1K_V1
            self.model = convnext_base(weights=weights)
            self.mean = weights.meta["mean"]
            self.std = weights.meta["std"]

        elif pretrained and convext_ver == "tiny":
            weights = ConvNeXt_Tiny_Weights.IMAGENET1K_V1
            self.model = convnext_tiny(weights=weights)
            self.mean = weights.meta["mean"]
            self.std = weights.meta["std"]

        else:
            self.model = convnext_large(weights=None)
            self.mean = [0.5, 0.5, 0.5]
            self.std = [0.5, 0.5, 0.5]

        # Replace the classifier for your number of classes
        in_features = self.model.classifier[2].in_features
        self.model.classifier[2] = nn.Linear(in_features, num_classes)  # type: ignore

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.

        Args:
            x (Tensor): Input tensor of shape [B, 3, 160, 160].

        Returns:
            Tensor: Output logits of shape [B, num_classes].
        """
        x = F.interpolate(x, size=(224, 224), mode="bilinear", align_corners=False)
        return self.model(x)
