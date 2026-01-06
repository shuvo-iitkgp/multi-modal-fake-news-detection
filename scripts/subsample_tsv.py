import argparse
import pandas as pd

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in_tsv", required=True)
    ap.add_argument("--out_tsv", required=True)
    ap.add_argument("--n", type=int, default=10000)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--label_col", default="2_way_label")  # keep distribution
    args = ap.parse_args()

    df = pd.read_csv(args.in_tsv, sep="\t")

    if args.label_col in df.columns:
        out = (
            df.groupby(args.label_col, group_keys=False)
              .apply(lambda x: x.sample(n=min(len(x), max(1, int(args.n * len(x) / len(df)))),
                                        random_state=args.seed))
        )
        # If rounding didn’t hit exactly n, top up/down
        if len(out) > args.n:
            out = out.sample(n=args.n, random_state=args.seed)
        elif len(out) < args.n:
            rem = df.drop(out.index)
            need = args.n - len(out)
            if need > 0 and len(rem) > 0:
                out = pd.concat([out, rem.sample(n=min(need, len(rem)), random_state=args.seed)], ignore_index=True)
    else:
        out = df.sample(n=min(args.n, len(df)), random_state=args.seed)

    out.to_csv(args.out_tsv, sep="\t", index=False)
    print(f"saved {len(out)} rows -> {args.out_tsv}")

if __name__ == "__main__":
    main()
