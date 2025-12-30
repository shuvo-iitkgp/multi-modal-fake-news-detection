from __future__ import annotations
import torch
import torch.nn as nn
import torch.nn.functional as F

from .text.bert import TextEncoder
from .vision.resnet50 import VisionEncoder
from .fusion.hierarchical_attention import HierarchicalAttentionFusion
from .heads.classifier import ClassifierHead
from .adversary.gradient_reversal import grad_reverse
from .adversary.event_discriminator import EventDiscriminator

class MMFNDModel(nn.Module):
    def __init__(
        self,
        text_model: str,
        max_len: int,
        use_image: bool,
        fused_dim: int,
        num_classes: int,
        use_event_adversary: bool = False,
        num_events: int = 0,
        grl_lambda: float = 0.5,
    ):
        super().__init__()
        self.text = TextEncoder(text_model, max_len=max_len)

        self.use_image = use_image
        if use_image:
            self.vision = VisionEncoder(pretrained=True)
            self.fusion = HierarchicalAttentionFusion(self.text.out_dim, self.vision.out_dim, fused_dim=fused_dim)
            feat_dim = fused_dim
        else:
            self.vision = None
            self.fusion = None
            feat_dim = self.text.out_dim

        self.head = ClassifierHead(feat_dim, num_classes=num_classes)

        self.use_event_adversary = use_event_adversary
        self.grl_lambda = grl_lambda
        if use_event_adversary:
            if num_events <= 1:
                raise ValueError("num_events must be >= 2 if use_event_adversary is enabled.")
            self.event_disc = EventDiscriminator(feat_dim, num_events=num_events)
        else:
            self.event_disc = None

    def forward(
        self,
        texts: list[str],
        images: torch.Tensor | list | None,
        device: torch.device,
        event_labels: torch.Tensor | None = None,
    ):
        t = self.text(texts, device=device)

        attn_w = None
        if self.use_image:
            # images can be stacked tensor or list with None
            if isinstance(images, torch.Tensor):
                v = self.vision(images.to(device))
            else:
                # missing images: replace with zeros
                B = len(texts)
                v = torch.zeros((B, self.vision.out_dim), device=device)
                for i, im in enumerate(images):
                    if im is not None:
                        v[i:i+1] = self.vision(im.unsqueeze(0).to(device))
            feat, attn_w = self.fusion(t, v)
        else:
            feat = t

        logits = self.head(feat)

        event_logits = None
        if self.use_event_adversary and event_labels is not None:
            rev = grad_reverse(feat, self.grl_lambda)
            event_logits = self.event_disc(rev)

        return logits, feat, attn_w, event_logits

def classification_loss(logits: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    return F.cross_entropy(logits, y)

def adversary_loss(event_logits: torch.Tensor, event_y: torch.Tensor) -> torch.Tensor:
    return F.cross_entropy(event_logits, event_y)
