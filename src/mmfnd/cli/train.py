from __future__ import annotations
import argparse
from pathlib import Path
import numpy as np
import torch

from mmfnd.config import load_config, deep_get
from mmfnd.common.seed import set_seed
from mmfnd.common.device import get_device
from mmfnd.common.paths import ensure_dir
from mmfnd.common.logging import setup_logger
from mmfnd.common.serialization import dump_json
from mmfnd.common.metrics import classification_metrics

from mmfnd.datasets.loaders import make_loader
from mmfnd.models.model import MMFNDModel
from mmfnd.train_utils import train_one_epoch, evaluate, build_event_vocab
from mmfnd.trust.domain_stats import build_domain_stats

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--outdir", default="outputs/runs/run1")
    ap.add_argument("--device", default=None)
    args = ap.parse_args()

    cfg = load_config(args.config)
    logger = setup_logger()

    seed = int(deep_get(cfg, "seed", 42))
    set_seed(seed)

    device = get_device(args.device)
    outdir = ensure_dir(args.outdir)
    dump_json(cfg, outdir / "config.json")

    # data
    train_csv = deep_get(cfg, "data.train_csv")
    val_csv = deep_get(cfg, "data.val_csv")
    image_root = deep_get(cfg, "data.image_root", None)
    image_size = int(deep_get(cfg, "data.image_size", 224))

    bs = int(deep_get(cfg, "train.batch_size", 8))
    nw = int(deep_get(cfg, "train.num_workers", 2))

    train_loader = make_loader(train_csv, batch_size=bs, shuffle=True, num_workers=nw, image_root=image_root, image_size=image_size)
    val_loader = make_loader(val_csv, batch_size=bs, shuffle=False, num_workers=nw, image_root=image_root, image_size=image_size)

    # Build event vocab from training samples for adversary
    # We need access to samples: easiest is to read one epoch worth quickly
    train_samples = []
    for b in train_loader:
        train_samples.extend(b["samples"])
    event_vocab = build_event_vocab(train_samples)

    # domain stats (for DTF later)
    domain_stats = build_domain_stats(train_samples)
    dump_json(domain_stats, outdir / "domain_stats.json")

    # model
    text_model = deep_get(cfg, "model.text_model", "distilbert-base-uncased")
    max_len = int(deep_get(cfg, "model.max_len", 256))
    use_image = bool(deep_get(cfg, "model.use_image", True))
    fused_dim = int(deep_get(cfg, "model.fused_dim", 512))
    num_classes = int(deep_get(cfg, "model.num_classes", 2))

    use_adv = bool(deep_get(cfg, "model.use_event_adversary", False))
    grl_lambda = float(deep_get(cfg, "model.grl_lambda", 0.5))
    adv_weight = float(deep_get(cfg, "model.adv_weight", 0.2))

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

    lr = float(deep_get(cfg, "train.lr", 2e-5))
    wd = float(deep_get(cfg, "train.weight_decay", 0.01))
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=wd)

    epochs = int(deep_get(cfg, "train.epochs", 3))
    best_f1 = -1.0
    best_path = outdir / "best.pt"

    for ep in range(1, epochs + 1):
        tr_loss = train_one_epoch(model, train_loader, opt, device, use_event_adversary=use_adv, event_vocab=event_vocab, adv_weight=adv_weight)
        val = evaluate(model, val_loader, device)
        m = classification_metrics(val["y"], val["yhat"])

        logger.info(f"epoch {ep} | train_loss={tr_loss:.4f} | val_macro_f1={m.macro_f1:.4f} | val_acc={m.accuracy:.4f}")

        dump_json({"epoch": ep, "train_loss": tr_loss, "val": {"accuracy": m.accuracy, "macro_f1": m.macro_f1}}, outdir / f"metrics_ep{ep}.json")

        if m.macro_f1 > best_f1:
            best_f1 = m.macro_f1
            torch.save({"model": model.state_dict(), "event_vocab": event_vocab, "cfg": cfg}, best_path)

    logger.info(f"best_macro_f1={best_f1:.4f}")
    logger.info(f"saved: {best_path}")

if __name__ == "__main__":
    main()
