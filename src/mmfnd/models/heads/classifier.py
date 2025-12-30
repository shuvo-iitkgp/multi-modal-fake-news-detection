import torch.nn as nn

class ClassifierHead(nn.Module):
    def __init__(self, in_dim: int, num_classes: int = 2, dropout: float = 0.1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(in_dim, num_classes),
        )

    def forward(self, x):
        return self.net(x)
