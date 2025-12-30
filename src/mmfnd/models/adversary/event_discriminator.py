import torch
import torch.nn as nn

class EventDiscriminator(nn.Module):
    def __init__(self, in_dim: int, num_events: int, hidden: int = 256, dropout: float = 0.1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, num_events),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)
