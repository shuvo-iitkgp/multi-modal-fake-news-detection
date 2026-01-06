import argparse
import pandas as pd
from sklearn.model_selection import train_test_split

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", required=True)
    ap.add_argument("--out_train", required=True)
    ap.add_argument("--out_val", required=True)
    ap.add_argument("--out_test", required=True)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--val_ratio", type=float, default=0.1)
    ap.add_argument("--test_ratio", type=float, default=0.1)
    args = ap.parse_args()

    df = pd.read_csv(args.csv)
    train_df, test_df = train_test_split(df, test_size=args.test_ratio, random_state=args.seed, stratify=df["label"])
    train_df, val_df = train_test_split(train_df, test_size=args.val_ratio / (1 - args.test_ratio),
                                        random_state=args.seed, stratify=train_df["label"])
    train_df.to_csv(args.out_train, index=False)
    val_df.to_csv(args.out_val, index=False)
    test_df.to_csv(args.out_test, index=False)
    print("saved splits")

if __name__ == "__main__":
    main()
