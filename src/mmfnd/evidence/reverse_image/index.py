from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path
from typing import List, Dict, Any, Optional

import numpy as np
import torch
from sklearn.neighbors import NearestNeighbors

from mmfnd.common.serialization import dump_json, load_json
from .embedder import ImageEmbedder


@dataclass
class ImageIndex:
    ids: List[str]
    feats: np.ndarray  # [N,D]
    nn: NearestNeighbors

    def save(self, outdir: str | Path) -> None:
        outdir = Path(outdir)
        outdir.mkdir(parents=True, exist_ok=True)
        np.save(outdir / "feats.npy", self.feats)
        dump_json({"ids": self.ids}, outdir / "meta.json")

    @staticmethod
    def load(indir: str | Path) -> "ImageIndex":
        indir = Path(indir)
        feats = np.load(indir / "feats.npy")
        meta = load_json(indir / "meta.json")
        ids = meta["ids"]
        nn = NearestNeighbors(metric="cosine", algorithm="auto")
        nn.fit(feats)
        return ImageIndex(ids=ids, feats=feats, nn=nn)


def build_image_index(
    samples: List[dict],
    device: torch.device,
    image_size: int,
    outdir: str | Path,
) -> None:
    """
    samples: list of sample dicts containing id and image_path
    """
    emb = ImageEmbedder(device=device, image_size=image_size)
    ids: List[str] = []
    vecs: List[np.ndarray] = []

    for s in samples:
        v = emb.embed_path(s.get("image_path", None))
        if v is None:
            continue
        ids.append(s["id"])
        vecs.append(v.numpy())

    if not vecs:
        raise RuntimeError("No valid images found for image index.")

    feats = np.stack(vecs, axis=0).astype(np.float32)
    nn = NearestNeighbors(metric="cosine", algorithm="auto")
    nn.fit(feats)

    idx = ImageIndex(ids=ids, feats=feats, nn=nn)
    idx.save(outdir)
