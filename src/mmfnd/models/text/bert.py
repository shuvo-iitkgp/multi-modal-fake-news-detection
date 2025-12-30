import torch
import torch.nn as nn
from transformers import AutoModel, AutoTokenizer

class TextEncoder(nn.Module):
    def __init__(self, model_name: str, max_len: int = 256):
        super().__init__()
        self.model_name = model_name
        self.max_len = max_len
        self.tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True)
        self.encoder = AutoModel.from_pretrained(model_name)

    @property
    def out_dim(self) -> int:
        return int(self.encoder.config.hidden_size)

    def forward(self, texts: list[str], device: torch.device) -> torch.Tensor:
        toks = self.tokenizer(
            texts,
            padding=True,
            truncation=True,
            max_length=self.max_len,
            return_tensors="pt",
        ).to(device)
        out = self.encoder(**toks)
        # CLS pooling
        cls = out.last_hidden_state[:, 0, :]
        return cls
