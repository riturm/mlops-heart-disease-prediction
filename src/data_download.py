"""
data_download.py
----------------
Produce a clean, headed CSV (data/heart_raw.csv) from the Heart Disease UCI data.

Two sources are supported:
  1. local : read the UCI file `processed.cleveland.data` that ships with the
             official archive. It has 14 comma-separated columns and NO header.
  2. uci   : fetch fresh via the `ucimlrepo` package (dataset id=45).

Run (recommended, since you already have the file):
    python src/data_download.py --source local \
        --in data/processed.cleveland.data --out data/heart_raw.csv
Or:
    python src/data_download.py --source uci --out data/heart_raw.csv
"""
import argparse
from pathlib import Path

import pandas as pd

# Canonical column order for processed Cleveland data (see heart-disease.names).
COLUMN_NAMES = [
    "age", "sex", "cp", "trestbps", "chol", "fbs", "restecg",
    "thalach", "exang", "oldpeak", "slope", "ca", "thal", "num",
]


def from_local(in_path: str) -> pd.DataFrame:
    """Read the header-less processed.cleveland.data and attach column names."""
    df = pd.read_csv(in_path, header=None, names=COLUMN_NAMES)
    return df


def from_uci() -> pd.DataFrame:
    from ucimlrepo import fetch_ucirepo
    heart = fetch_ucirepo(id=45)
    df = pd.concat([heart.data.features, heart.data.targets], axis=1)
    df = df.rename(columns={df.columns[-1]: "num"})
    return df


def main(source: str, in_path: str, out_path: str) -> None:
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    df = from_local(in_path) if source == "local" else from_uci()
    df.to_csv(out_path, index=False)
    print(f"[data_download] source={source}  saved "
          f"{df.shape[0]} rows x {df.shape[1]} cols -> {out_path}")
    print(df.head())


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--source", choices=["local", "uci"], default="local")
    p.add_argument("--in", dest="in_path", default="data/processed.cleveland.data")
    p.add_argument("--out", default="data/heart_raw.csv")
    args = p.parse_args()
    main(args.source, args.in_path, args.out)
