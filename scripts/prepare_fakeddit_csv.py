import argparse
import os
import pandas as pd

def pick_first(df, candidates):
    for c in candidates:
        if c in df.columns:
            return c
    return None

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tsv", required=True)
    ap.add_argument("--out_csv", required=True)
    ap.add_argument("--image_root", default=None, help="prefix for relative image paths")
    ap.add_argument("--id_col", default=None)
    ap.add_argument("--text_col", default=None)
    ap.add_argument("--image_col", default=None)
    ap.add_argument("--label_col", default=None)
    ap.add_argument("--event_col", default=None, help="optional; becomes event_id")
    ap.add_argument("--domains_col", default=None, help="optional; should be JSON list or 'a|b|c'")
    args = ap.parse_args()

    df = pd.read_csv(args.tsv, sep="\t")

    id_col = args.id_col or pick_first(df, ["id", "post_id", "submission_id"])
    text_col = args.text_col or pick_first(df, ["clean_title", "title", "text", "body", "caption"])
    image_col = args.image_col or pick_first(df, ["image_path", "img_path", "image", "img"])
    label_col = args.label_col or pick_first(df, ["2_way_label", "label", "class", "y"])

    if id_col is None:
        df["id"] = [str(i) for i in range(len(df))]
        id_col = "id"
    if text_col is None:
        raise ValueError("Could not find text column. Pass --text_col.")
    if image_col is None:
        # image-only and multimodal will fail if you don't have images
        print("WARNING: no image column found. Image-only and multimodal will not work.")
    if label_col is None:
        raise ValueError("Could not find label column. Pass --label_col.")

    out = pd.DataFrame()
    out["id"] = df[id_col].astype(str)
    out["text"] = df[text_col].fillna("").astype(str)

    if image_col is not None:
        img = df[image_col].fillna("").astype(str)
        if args.image_root:
            out["image_path"] = img.apply(lambda p: os.path.join(args.image_root, p) if p and not os.path.isabs(p) else p)
        else:
            out["image_path"] = img
    else:
        out["image_path"] = ""

    # label must be int for your training loop
    # if it's not already, factorize
    if pd.api.types.is_integer_dtype(df[label_col]):
        out["label"] = df[label_col].astype(int)
    else:
        codes, _ = pd.factorize(df[label_col].astype(str))
        out["label"] = codes.astype(int)

    # event_id for adversary (optional)
    if args.event_col and args.event_col in df.columns:
        out["event_id"] = df[args.event_col].fillna("NO_EVENT").astype(str)
    else:
        out["event_id"] = "NO_EVENT"

    # domains used for DTF scoring (optional). keep as pipe-separated string.
    if args.domains_col and args.domains_col in df.columns:
        out["domains"] = df[args.domains_col].fillna("").astype(str)
    else:
        out["domains"] = ""

    out.to_csv(args.out_csv, index=False)
    print(f"saved: {args.out_csv}")

if __name__ == "__main__":
    main()
