from __future__ import annotations
from dataclasses import asdict
from pathlib import Path
from typing import Any, Dict, Optional

import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from PIL import Image

from .schemas import Sample
from .text_clean import extract_urls, domains_from_urls, normalize_text
from .image_transforms import default_image_transform

class CSVDataset(Dataset):
    """
    Expected columns:
      id, text, image_path, label, lang, event_id, topic_id
    label can be missing for test.
    """
    def __init__(self, csv_path: str | Path, image_root: Optional[str | Path] = None, image_size: int = 224):
        self.df = pd.read_csv(csv_path)
        self.image_root = Path(image_root) if image_root else None
        self.tf = default_image_transform(image_size=image_size)

        # normalize text and pre-extract domains
        self.df["text"] = self.df["text"].fillna("").map(normalize_text)
        self.df["urls"] = self.df["text"].map(extract_urls)
        self.df["domains"] = self.df["urls"].map(domains_from_urls)

    def __len__(self) -> int:
        return len(self.df)

    def _resolve_image(self, p: Any) -> Optional[str]:
        if pd.isna(p) or p is None or str(p).strip() == "":
            return None
        p = Path(str(p))
        if self.image_root and not p.is_absolute():
            p = self.image_root / p
        return str(p)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        row = self.df.iloc[idx]
        img_path = self._resolve_image(row.get("image_path", None))

        image_tensor = None
        if img_path is not None and Path(img_path).exists():
            img = Image.open(img_path).convert("RGB")
            image_tensor = self.tf(img)

        label = row.get("label", None)
        if pd.isna(label):
            label = None
        else:
            label = int(label)

        sample = Sample(
            id=str(row.get("id", idx)),
            text=str(row.get("text", "")),
            image_path=img_path,
            label=label,
            lang=str(row.get("lang", "")) if "lang" in row else None,
            urls=list(row.get("urls", [])),
            domains=list(row.get("domains", [])),
            event_id=str(row.get("event_id", "")) if "event_id" in row else None,
            topic_id=str(row.get("topic_id", "")) if "topic_id" in row else None,
            meta={},
        )

        return {
            "sample": asdict(sample),
            "image": image_tensor,  # can be None
        }

def collate_fn(batch):
    # batch is list of dicts {sample, image}
    samples = [b["sample"] for b in batch]
    images = [b["image"] for b in batch]
    # stack images if all exist, else keep None markers
    if all(img is not None for img in images):
        images = torch.stack(images, dim=0)
    else:
        images = images
    return {"samples": samples, "images": images}

def make_loader(csv_path: str, batch_size: int, shuffle: bool, num_workers: int, image_root: str | None, image_size: int):
    ds = CSVDataset(csv_path=csv_path, image_root=image_root, image_size=image_size)
    return DataLoader(ds, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers, collate_fn=collate_fn)
