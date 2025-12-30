from __future__ import annotations
import argparse
import json
from pathlib import Path
import torch

from mmfnd.common.device import get_device
from mmfnd.common.logging import setup_logger
from mmfnd.common.paths import ensure_dir
from mmfnd.datasets.loaders import make_loader
from mmfnd.evidence.reverse_image.index import build_image_index
from mmfnd.evidence.dpr.index import build_text_index


def read_samples_from_loader(loader, limit: int | None = None):
    out = []
    for b in loader:
        out.extend(b["samples"])
        if limit is not None and len(out) >= limit:
            return out[:limit]
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--samples_csv", required=True, help="CSV with id,text,image_path,label,...")
    ap.add_argument("--image_root", default=None)
    ap.add_argument("--image_size", type=int, default=224)
    ap.add_argument("--batch_size", type=int, default=16)
    ap.add_argument("--num_workers", type=int, default=2)
    ap.add_argument("--outdir", default="data/indices")
    ap.add_argument("--docs_jsonl", required=True, help="Local corpus JSONL: {doc_id,title,text}")
    ap.add_argument("--device", default=None)
    args = ap.parse_args()

    logger = setup_logger()
    device = get_device(args.device)
    outdir = ensure_dir(args.outdir)

    loader = make_loader(
        args.samples_csv,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        image_root=args.image_root,
        image_size=args.image_size,
    )
    samples = read_samples_from_loader(loader)
    logger.info(f"loaded samples: {len(samples)}")

    # build image index
    img_out = outdir / "reverse_image"
    logger.info(f"building image index -> {img_out}")
    build_image_index(samples=samples, device=device, image_size=args.image_size, outdir=img_out)

    # build text index
    docs = []
    with open(args.docs_jsonl, "r", encoding="utf-8") as f:
        for line in f:
            docs.append(json.loads(line))
    txt_out = outdir / "text_retrieval"
    logger.info(f"building text index -> {txt_out} | docs={len(docs)}")
    build_text_index(docs=docs, outdir=txt_out)

    logger.info("done")

if __name__ == "__main__":
    main()
