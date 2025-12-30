from __future__ import annotations
from dataclasses import dataclass
from typing import Dict, Any, Optional
import time
import numpy as np
import torch
from tqdm import tqdm

from mmfnd.models.model import classification_loss, adversary_loss

@dataclass
class TrainState:
    step: int = 0
    epoch: int = 0

def to_event_index(event_ids: list[str], vocab: dict[str, int]) -> torch.Tensor:
    idx = [vocab.get(e or "NO_EVENT", vocab["NO_EVENT"]) for e in event_ids]
    return torch.tensor(idx, dtype=torch.long)

def build_event_vocab(samples: list[dict]) -> dict[str, int]:
    events = ["NO_EVENT"]
    for s in samples:
        e = s.get("event_id", None) or "NO_EVENT"
        if e not in events:
            events.append(e)
    return {e: i for i, e in enumerate(events)}

@torch.no_grad()
def evaluate(model, loader, device) -> Dict[str, Any]:
    model.eval()
    ys, preds, probs = [], [], []
    t0 = time.time()
    for batch in loader:
        samples = batch["samples"]
        texts = [s["text"] for s in samples]
        images = batch["images"]
        logits, _, _, _ = model(texts, images, device=device, event_labels=None)
        p = torch.softmax(logits, dim=-1)[:, 1].detach().cpu().numpy()
        y = np.array([int(s["label"]) for s in samples], dtype=int)
        yhat = (p >= 0.5).astype(int)
        ys.append(y); preds.append(yhat); probs.append(p)
    dt = time.time() - t0
    ys = np.concatenate(ys); preds = np.concatenate(preds); probs = np.concatenate(probs)
    return {"y": ys, "yhat": preds, "p_real": probs, "seconds": dt}

def train_one_epoch(
    model,
    loader,
    opt,
    device,
    use_event_adversary: bool,
    event_vocab: Optional[dict[str, int]] = None,
    adv_weight: float = 0.2,
):
    model.train()
    losses = []
    for batch in tqdm(loader, desc="train", leave=False):
        samples = batch["samples"]
        texts = [s["text"] for s in samples]
        images = batch["images"]
        y = torch.tensor([int(s["label"]) for s in samples], dtype=torch.long, device=device)

        event_y = None
        if use_event_adversary:
            if event_vocab is None:
                raise ValueError("event_vocab required when use_event_adversary=True")
            event_ids = [s.get("event_id", None) or "NO_EVENT" for s in samples]
            event_y = to_event_index(event_ids, event_vocab).to(device)

        opt.zero_grad(set_to_none=True)
        logits, feat, _, event_logits = model(texts, images, device=device, event_labels=event_y)

        loss = classification_loss(logits, y)
        if use_event_adversary and event_logits is not None and event_y is not None:
            loss = loss + adv_weight * adversary_loss(event_logits, event_y)

        loss.backward()
        opt.step()
        losses.append(float(loss.detach().cpu().item()))
    return float(np.mean(losses)) if losses else 0.0
