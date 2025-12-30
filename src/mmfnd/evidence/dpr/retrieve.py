from __future__ import annotations
from pathlib import Path
from typing import Any, Dict, List

import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

from .index import TextIndex


class TextRetriever:
    def __init__(self, index_dir: str | Path):
        self.index = TextIndex.load(index_dir)

    def query(self, query_text: str, topk: int = 5) -> List[Dict[str, Any]]:
        if not query_text or not query_text.strip():
            return []
        q = self.index.vectorizer.transform([query_text])
        sims = cosine_similarity(q, self.index.X).reshape(-1)  # [N]
        k = min(topk, sims.shape[0])
        idxs = np.argsort(-sims)[:k]

        out = []
        for i in idxs:
            out.append({
                "doc_id": self.index.doc_ids[int(i)],
                "title": self.index.titles[int(i)],
                "text": self.index.texts[int(i)],
                "score": float(sims[int(i)]),
            })
        return out
