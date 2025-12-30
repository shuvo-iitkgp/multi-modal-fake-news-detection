from __future__ import annotations
import argparse
from pathlib import Path
import random
import json
import pandas as pd
import requests
from datasets import load_dataset
from tqdm import tqdm

def safe_download(url: str, out_path: Path, timeout: int = 20) -> bool:
    try:
        r = requests.get(url, timeout=timeout)
        r.raise_for_status()
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_bytes(r.content)
        return True
    except Exception:
        return False

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out_dir", default="data/fakeddit_subset")
    ap.add_argument("--n_train", type=int, default=500)
    ap.add_argument("--n_val", type=int, default=200)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--download_images", action="store_true")
    args = ap.parse_args()

    random.seed(args.seed)
    out_dir = Path(args.out_dir)
    img_dir = out_dir / "images"
    out_dir.mkdir(parents=True, exist_ok=True)

    # Load Fakeddit
    ds = load_dataset("fakeddit/fakeddit", "all")  # relies on HF datasets hub
    # Use the 2-way label for baseline: 0/1
    train = ds["train"]
    val = ds["validation"] if "validation" in ds else ds["test"]

    def sample_rows(split, n):
        idxs = list(range(len(split)))
        random.shuffle(idxs)
        idxs = idxs[:n]
        rows = []
        for i in idxs:
            ex = split[int(i)]
            # fields can vary; use robust access
            _id = str(ex.get("id", i))
            text = ex.get("clean_title", None) or ex.get("title", None) or ex.get("text", "") or ""
            label = ex.get("2_way_label", None)
            if label is None:
                continue
            # image url field varies, try common keys
            img_url = ex.get("image_url", None) or ex.get("image", None) or ex.get("url", None)
            rows.append({"id": _id, "text": text, "label": int(label), "image_url": img_url})
        return rows

    train_rows = sample_rows(train, args.n_train * 2)
    val_rows = sample_rows(val, args.n_val * 2)

    # Keep only rows with text
    train_rows = [r for r in train_rows if r["text"].strip()][:args.n_train]
    val_rows = [r for r in val_rows if r["text"].strip()][:args.n_val]

    def write_csv(rows, name):
        df = pd.DataFrame(rows)
        df["image_path"] = ""
        if args.download_images:
            kept = []
            for r in tqdm(rows, desc=f"download {name} imgs"):
                url = r.get("image_url", None)
                if not url or not isinstance(url, str) or not url.startswith("http"):
                    kept.append(r)
                    continue
                out_path = img_dir / f"{r['id']}.jpg"
                ok = safe_download(url, out_path)
                if ok:
                    r["image_path"] = out_path.relative_to(out_dir).as_posix().replace("images/", "")
                    # we want image_root to be out_dir/images, so store just filename
                    r["image_path"] = f"{r['id']}.jpg"
                kept.append(r)
            df = pd.DataFrame(kept)
        df.to_csv(out_dir / f"{name}.csv", index=False)

    write_csv(train_rows, "train")
    write_csv(val_rows, "val")

    meta = {"out_dir": str(out_dir), "n_train": len(train_rows), "n_val": len(val_rows), "seed": args.seed}
    (out_dir / "meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    print("Saved:", meta)

if __name__ == "__main__":
    main()
