import torch
import torch.nn as nn
from torchvision.models import resnet50, ResNet50_Weights

class VisionEncoder(nn.Module):
    def __init__(self, pretrained: bool = True):
        super().__init__()
        weights = ResNet50_Weights.DEFAULT if pretrained else None
        m = resnet50(weights=weights)
        self.backbone = nn.Sequential(*list(m.children())[:-1])  # up to avgpool
        self._out_dim = 2048

    @property
    def out_dim(self) -> int:
        return self._out_dim

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        x = self.backbone(images)  # [B, 2048, 1, 1]
        x = x.flatten(1)
        return x
