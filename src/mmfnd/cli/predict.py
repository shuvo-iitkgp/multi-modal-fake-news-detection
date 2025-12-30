from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from mmfnd.config import load_config, deep_get
from mmfnd.common.device import get_device
from mmfnd.common.serialization import load_json
from mmfnd.datasets.loaders import make_loader
from mmfnd.models.model import MMFNDModel
from mmfnd.trust.dtf import dtf_for_sample
from mmfnd.scoring.cmpu import cmpu
from mmfnd.scoring.triage import rank_for_verification
from mmfnd.evidence.cache import load_evidence_jsonl


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--csv", required=True)
    ap.add_argument("--domain_stats", required=True)
    ap.add_argument("--out_csv", default="outputs/predictions.csv")
    ap.add_argument("--device", default=None)
    args = ap.parse_args()

    cfg = load_config(args.config)
    device = get_device(args.device)
    domain_stats = load_json(args.domain_stats)

    image_root = deep_get(cfg, "data.image_root", None)
    image_size = int(deep_get(cfg, "data.image_size", 224))
    bs = int(deep_get(cfg, "train.batch_size", 8))
    nw = int(deep_get(cfg, "train.num_workers", 2))

    loader = make_loader(
        args.csv,
        batch_size=bs,
        shuffle=False,
        num_workers=nw,
        image_root=image_root,
        image_size=image_size,
    )

    text_model = deep_get(cfg, "model.text_model", "distilbert-base-uncased")
    max_len = int(deep_get(cfg, "model.max_len", 256))
    use_image = bool(deep_get(cfg, "model.use_image", True))
    fused_dim = int(deep_get(cfg, "model.fused_dim", 512))
    num_classes = int(deep_get(cfg, "model.num_classes", 2))
    use_adv = bool(deep_get(cfg, "model.use_event_adversary", False))
    grl_lambda = float(deep_get(cfg, "model.grl_lambda", 0.5))

    if num_classes != 2:
        raise ValueError("predict.py currently assumes binary classification (num_classes=2).")

    ckpt = torch.load(args.ckpt, map_location="cpu")
    event_vocab = ckpt.get("event_vocab", {"NO_EVENT": 0})

    model = MMFNDModel(
        text_model=text_model,
        max_len=max_len,
        use_image=use_image,
        fused_dim=fused_dim,
        num_classes=num_classes,
        use_event_adversary=use_adv,
        num_events=len(event_vocab),
        grl_lambda=grl_lambda,
    ).to(device)
    model.load_state_dict(ckpt["model"], strict=True)
    model.eval()

    ids: list[str] = []
    probs: list[float] = []
    dtfs: list[float] = []

    with torch.no_grad():
        for batch in loader:
            samples = batch["samples"]
            texts = [s["text"] for s in samples]
            images = batch["images"]

            ids.extend([s["id"] for s in samples])

            logits, _, _, _ = model(texts, images, device=device, event_labels=None)

            # force 1-D [B]
            p = torch.softmax(logits, dim=-1)[:, 1].detach().cpu().numpy().reshape(-1)
            probs.extend(p.tolist())

            for s in samples:
                dtfs.append(dtf_for_sample(s.get("domains", []), domain_stats))

    probs_arr = np.array(probs, dtype=float)
    dtfs_arr = np.array(dtfs, dtype=float)

    beta = float(deep_get(cfg, "scoring.beta_cmpu", 0.5))
    cmpu_scores = cmpu(probs_arr, dtfs_arr, beta=beta)

    # optional evidence-aware CMPU
    ev_path = deep_get(cfg, "evidence.jsonl", None)
    use_ver = bool(deep_get(cfg, "evidence.use_verifier_in_scoring", False))
    ver_w = float(deep_get(cfg, "evidence.verifier_weight", 0.3))

    ver_scores = np.zeros_like(probs_arr)
    if use_ver and ev_path and Path(ev_path).exists():
        ev_map = load_evidence_jsonl(ev_path)
        for i, _id in enumerate(ids):
            ver_scores[i] = float(ev_map.get(str(_id), {}).get("verifier", {}).get("score", 0.0))
        cmpu_scores = cmpu_scores + ver_w * (1.0 - ver_scores)

    ranking = rank_for_verification(cmpu_scores)

    df = pd.DataFrame({
        "id": ids,
        "p_real": probs_arr,
        "dtf": dtfs_arr,
        "cmpu": cmpu_scores,
        "triage_rank": np.empty(len(ids), dtype=int),
    })
    df.iloc[ranking, df.columns.get_loc("triage_rank")] = np.arange(len(ids))

    Path(args.out_csv).parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(args.out_csv, index=False)
    print(f"saved: {args.out_csv}")


if __name__ == "__main__":
    main()
