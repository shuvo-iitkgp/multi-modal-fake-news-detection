import torch
import torch.nn as nn
import torch.nn.functional as F

class HierarchicalAttentionFusion(nn.Module):
    """
    Minimal: produce modality weights and fuse embeddings.
    Inputs: text_feat [B, Dt], img_feat [B, Di]
    Output: fused [B, D]
    """
    def __init__(self, text_dim: int, img_dim: int, fused_dim: int = 512, dropout: float = 0.1):
        super().__init__()
        self.text_proj = nn.Linear(text_dim, fused_dim)
        self.img_proj = nn.Linear(img_dim, fused_dim)
        self.attn = nn.Sequential(
            nn.Linear(fused_dim * 2, fused_dim),
            nn.Tanh(),
            nn.Linear(fused_dim, 2),  # weights for (text, image)
        )
        self.dropout = nn.Dropout(dropout)

    def forward(self, text_feat: torch.Tensor, img_feat: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        t = self.dropout(self.text_proj(text_feat))
        v = self.dropout(self.img_proj(img_feat))
        h = torch.cat([t, v], dim=-1)
        logits = self.attn(h)
        w = F.softmax(logits, dim=-1)  # [B,2]
        fused = w[:, :1] * t + w[:, 1:] * v
        return fused, w
