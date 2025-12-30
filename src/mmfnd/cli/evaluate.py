from __future__ import annotations
import argparse
from pathlib import Path
import numpy as np
import torch

from mmfnd.config import load_config, deep_get
from mmfnd.common.device import get_device
from mmfnd.common.logging import setup_logger
from mmfnd.common.serialization import load_json, dump_json
from mmfnd.common.metrics import classification_metrics
from mmfnd.datasets.loaders import make_loader
from mmfnd.models.model import MMFNDModel
from mmfnd.trust.dtf import dtf_for_sample
from mmfnd.scoring.pts import pts
from mmfnd.scoring.cmc import cmc
from mmfnd.scoring.cmpu import cmpu
from mmfnd.scoring.triage import rank_for_verification
from mmfnd.evidence.cache import load_evidence_jsonl

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--split_csv", required=True)  # val or test
    ap.add_argument("--domain_stats", required=True)
    ap.add_argument("--out", default="outputs/eval.json")
    ap.add_argument("--device", default=None)
    args = ap.parse_args()

    logger = setup_logger()
    cfg = load_config(args.config)
    device = get_device(args.device)

    domain_stats = load_json(args.domain_stats)

    image_root = deep_get(cfg, "data.image_root", None)
    image_size = int(deep_get(cfg, "data.image_size", 224))
    bs = int(deep_get(cfg, "train.batch_size", 8))
    nw = int(deep_get(cfg, "train.num_workers", 2))

    loader = make_loader(args.split_csv, batch_size=bs, shuffle=False, num_workers=nw, image_root=image_root, image_size=image_size)

    text_model = deep_get(cfg, "model.text_model", "distilbert-base-uncased")
    max_len = int(deep_get(cfg, "model.max_len", 256))
    use_image = bool(deep_get(cfg, "model.use_image", True))
    fused_dim = int(deep_get(cfg, "model.fused_dim", 512))
    num_classes = int(deep_get(cfg, "model.num_classes", 2))
    use_adv = bool(deep_get(cfg, "model.use_event_adversary", False))
    grl_lambda = float(deep_get(cfg, "model.grl_lambda", 0.5))

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
 

    
    ys, yhat = [], []
    p_real, dtfs, ids = [], [], []

    with torch.no_grad():
        for batch in loader:
            samples = batch["samples"]
            texts = [s["text"] for s in samples]
            images = batch["images"]

            ids.extend([s["id"] for s in samples])

            logits, _, _, _ = model(texts, images, device=device, event_labels=None)

            # force consistent shape: [B]
            p = torch.softmax(logits, dim=-1)[:, 1].detach().cpu().numpy().reshape(-1)
            p_real.extend(p.tolist())

            for s in samples:
                dtfs.append(dtf_for_sample(s.get("domains", []), domain_stats))

            if samples[0].get("label", None) is not None:
                y = np.array([int(s["label"]) for s in samples], dtype=int)
                yh = (p >= 0.5).astype(int)
                ys.append(y)
                yhat.append(yh)

    p_real = np.array(p_real, dtype=float)
    dtfs = np.array(dtfs, dtype=float)
    ids = np.array(ids)



 

    out = {"n": int(len(p_real))}
    if ys:
        ys = np.concatenate(ys); yhat = np.concatenate(yhat)
        m = classification_metrics(ys, yhat)
        out["metrics"] = {"accuracy": m.accuracy, "macro_f1": m.macro_f1}

    # triage signals
    alpha = float(deep_get(cfg, "scoring.alpha_pts", 0.5))
    w_model = float(deep_get(cfg, "scoring.w_model_cmc", 0.7))
    beta = float(deep_get(cfg, "scoring.beta_cmpu", 0.5))
    ev_path = deep_get(cfg, "evidence.jsonl", None)
    use_ver = bool(deep_get(cfg, "evidence.use_verifier_in_scoring", False))
    ver_w = float(deep_get(cfg, "evidence.verifier_weight", 0.3))

    ver_scores = np.zeros_like(p_real)
    if ev_path and Path(ev_path).exists():
        ev_map = load_evidence_jsonl(ev_path)
        for i, _id in enumerate(ids):
            ver_scores[i] = float(ev_map.get(str(_id), {}).get("verifier", {}).get("score", 0.0))

    PTS = pts(p_real, dtfs, alpha=alpha)
    CMC = cmc(p_real, dtfs, w_model=w_model)
    CMPU = cmpu(p_real, dtfs, beta=beta)
    if use_ver:
        # If verifier says "supports", reduce uncertainty; otherwise increase it.
        CMPU = CMPU + ver_w * (1.0 - ver_scores)
    out["signals"] = {
        "pts_mean": float(PTS.mean()),
        "cmc_mean": float(CMC.mean()),
        "cmpu_mean": float(CMPU.mean()),
    }
    out["triage_top10_idx"] = rank_for_verification(CMPU)[:10].tolist()

    dump_json(out, args.out)
    logger.info(f"saved: {args.out}")
    logger.info(str(out))

if __name__ == "__main__":
    main()
