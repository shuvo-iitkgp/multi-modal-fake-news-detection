from __future__ import annotations
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

from mmfnd.evidence.reverse_image.retrieve import ReverseImageRetriever
from mmfnd.evidence.dpr.retrieve import TextRetriever
from mmfnd.evidence.coref.resolve import simple_coref_resolve
from mmfnd.evidence.verifier.t5_entailment import T5EntailmentVerifier


@dataclass
class EvidenceConfig:
    use_reverse_image: bool = True
    use_text_retrieval: bool = True
    use_coref: bool = True
    use_verifier: bool = True
    img_topk: int = 5
    text_topk: int = 5


class EvidencePipeline:
    def __init__(
        self,
        cfg: EvidenceConfig,
        img_retriever: Optional[ReverseImageRetriever],
        text_retriever: Optional[TextRetriever],
        verifier: Optional[T5EntailmentVerifier],
    ):
        self.cfg = cfg
        self.img_retriever = img_retriever
        self.text_retriever = text_retriever
        self.verifier = verifier

    def run_one(self, sample: dict) -> Dict[str, Any]:
        """
        sample keys expected:
          id, text, image_path (optional)
        """
        out: Dict[str, Any] = {"id": sample["id"]}

        text = sample.get("text", "") or ""

        # coref first to make retrieval queries cleaner
        if self.cfg.use_coref and self.text_retriever is not None:
            text_for_retrieval = simple_coref_resolve(text)
        else:
            text_for_retrieval = text

        # reverse-image (dataset-local nearest neighbors)
        if self.cfg.use_reverse_image and self.img_retriever is not None:
            img_path = sample.get("image_path", None)
            out["img_neighbors"] = self.img_retriever.query(img_path, topk=self.cfg.img_topk)
        else:
            out["img_neighbors"] = []

        # text retrieval (local corpus)
        if self.cfg.use_text_retrieval and self.text_retriever is not None:
            out["text_passages"] = self.text_retriever.query(text_for_retrieval, topk=self.cfg.text_topk)
        else:
            out["text_passages"] = []

        # verifier: “does evidence support claim?”
        if self.cfg.use_verifier and self.verifier is not None:
            # build a compact evidence string
            evidence_lines: List[str] = []
            for n in out["img_neighbors"][: min(3, len(out["img_neighbors"]))]:
                # neighbor text is optional
                t = n.get("text", "")
                if t:
                    evidence_lines.append(t)

            for p in out["text_passages"][: min(3, len(out["text_passages"]))]:
                evidence_lines.append(p.get("text", ""))

            evidence_text = "\n".join([e.strip() for e in evidence_lines if e and e.strip()][:6])
            out["verifier"] = self.verifier.score(claim=text, evidence=evidence_text)
        else:
            out["verifier"] = {"score": 0.0, "label": "UNKNOWN"}

        return out
