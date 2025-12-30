from __future__ import annotations
import argparse
import json
from pathlib import Path
import torch
from tqdm import tqdm

from mmfnd.common.device import get_device
from mmfnd.common.logging import setup_logger
from mmfnd.common.paths import ensure_dir
from mmfnd.datasets.loaders import make_loader

from mmfnd.evidence.pipeline import EvidencePipeline, EvidenceConfig
from mmfnd.evidence.reverse_image.retrieve import ReverseImageRetriever
from mmfnd.evidence.dpr.retrieve import TextRetriever
from mmfnd.evidence.verifier.t5_entailment import T5EntailmentVerifier


def read_samples_from_loader(loader):
    out = []
    for b in loader:
        out.extend(b["samples"])
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--samples_csv", required=True)
    ap.add_argument("--image_root", default=None)
    ap.add_argument("--image_size", type=int, default=224)
    ap.add_argument("--batch_size", type=int, default=8)
    ap.add_argument("--num_workers", type=int, default=2)

    ap.add_argument("--indices_dir", default="data/indices")
    ap.add_argument("--out_jsonl", default="data/cache/evidence.jsonl")

    ap.add_argument("--use_reverse_image", action="store_true")
    ap.add_argument("--use_text_retrieval", action="store_true")
    ap.add_argument("--use_coref", action="store_true")
    ap.add_argument("--use_verifier", action="store_true")

    ap.add_argument("--img_topk", type=int, default=5)
    ap.add_argument("--text_topk", type=int, default=5)

    ap.add_argument("--verifier_model", default="t5-small")
    ap.add_argument("--device", default=None)
    args = ap.parse_args()

    logger = setup_logger()
    device = get_device(args.device)

    loader = make_loader(
        args.samples_csv,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        image_root=args.image_root,
        image_size=args.image_size,
    )
    samples = read_samples_from_loader(loader)
    logger.info(f"samples: {len(samples)}")

    img_ret = None
    txt_ret = None
    ver = None

    if args.use_reverse_image:
        img_ret = ReverseImageRetriever(
            index_dir=Path(args.indices_dir) / "reverse_image",
            device=device,
            image_size=args.image_size,
        )

    if args.use_text_retrieval:
        txt_ret = TextRetriever(index_dir=Path(args.indices_dir) / "text_retrieval")

    if args.use_verifier:
        ver = T5EntailmentVerifier(model_name=args.verifier_model, device=device, max_len=256)

    cfg = EvidenceConfig(
        use_reverse_image=args.use_reverse_image,
        use_text_retrieval=args.use_text_retrieval,
        use_coref=args.use_coref,
        use_verifier=args.use_verifier,
        img_topk=args.img_topk,
        text_topk=args.text_topk,
    )

    pipe = EvidencePipeline(cfg=cfg, img_retriever=img_ret, text_retriever=txt_ret, verifier=ver)

    out_path = Path(args.out_jsonl)
    ensure_dir(out_path.parent)

    with open(out_path, "w", encoding="utf-8") as f:
        for s in tqdm(samples, desc="evidence"):
            ev = pipe.run_one(s)
            f.write(json.dumps(ev, ensure_ascii=False) + "\n")

    logger.info(f"saved evidence -> {out_path}")

if __name__ == "__main__":
    main()
