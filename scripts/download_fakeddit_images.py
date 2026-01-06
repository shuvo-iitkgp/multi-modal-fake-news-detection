import argparse, os, hashlib
import pandas as pd
import requests
from tqdm import tqdm

def fname_from_url(url: str) -> str:
    h = hashlib.md5(url.encode("utf-8")).hexdigest()
    return f"{h}.jpg"

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tsv", required=True)
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--url_col", default="image_url")
    ap.add_argument("--id_col", default="id")
    ap.add_argument("--out_map_csv", required=True, help="writes id->image_path mapping")
    ap.add_argument("--timeout", type=int, default=10)
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    df = pd.read_csv(args.tsv, sep="\t")

    rows = []
    for _, r in tqdm(df.iterrows(), total=len(df)):
        pid = str(r.get(args.id_col, ""))
        url = r.get(args.url_col, "")
        url = "" if pd.isna(url) else str(url).strip()

        rel_path = ""
        if url:
            rel_path = fname_from_url(url)
            out_path = os.path.join(args.out_dir, rel_path)
            if not os.path.exists(out_path):
                try:
                    resp = requests.get(url, timeout=args.timeout)
                    if resp.status_code == 200 and resp.content:
                        with open(out_path, "wb") as f:
                            f.write(resp.content)
                    else:
                        rel_path = ""
                except Exception:
                    rel_path = ""

        rows.append({"id": pid, "image_path": rel_path})

    pd.DataFrame(rows).to_csv(args.out_map_csv, index=False)
    print(f"saved mapping: {args.out_map_csv}")

if __name__ == "__main__":
    main()
