from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path
from typing import List, Dict, Any

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer

from mmfnd.common.serialization import dump_json, load_json


@dataclass
class TextIndex:
    doc_ids: List[str]
    texts: List[str]
    titles: List[str]
    vectorizer: TfidfVectorizer
    X: Any  # sparse matrix

    def save(self, outdir: str | Path) -> None:
        outdir = Path(outdir)
        outdir.mkdir(parents=True, exist_ok=True)
        # sklearn vectorizer is not trivially JSON; store vocab + idf
        dump_json(
            {
                "doc_ids": self.doc_ids,
                "titles": self.titles,
                "texts": self.texts,
                "vocab": self.vectorizer.vocabulary_,
                "idf": self.vectorizer.idf_.tolist(),
            },
            outdir / "text_index.json",
        )

    @staticmethod
    def load(indir: str | Path) -> "TextIndex":
        indir = Path(indir)
        obj = load_json(indir / "text_index.json")
        vec = TfidfVectorizer(stop_words="english")
        vec.vocabulary_ = {k: int(v) for k, v in obj["vocab"].items()}
        vec.idf_ = np.array(obj["idf"], dtype=np.float64)
        vec._tfidf._idf_diag = None  # will rebuild lazily
        texts = obj["texts"]
        X = vec.transform(texts)
        return TextIndex(
            doc_ids=obj["doc_ids"],
            titles=obj["titles"],
            texts=texts,
            vectorizer=vec,
            X=X,
        )


def build_text_index(docs: List[Dict[str, Any]], outdir: str | Path) -> None:
    """
    docs: list of {doc_id, title, text}
    """
    doc_ids = [str(d["doc_id"]) for d in docs]
    titles = [str(d.get("title", "")) for d in docs]
    texts = [str(d.get("text", "")) for d in docs]

    vec = TfidfVectorizer(stop_words="english", max_features=200000)
    X = vec.fit_transform(texts)

    idx = TextIndex(doc_ids=doc_ids, titles=titles, texts=texts, vectorizer=vec, X=X)
    idx.save(outdir)
