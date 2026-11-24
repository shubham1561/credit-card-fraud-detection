"""Load, validate and de-duplicate the ULB credit-card dataset.

    python -m src.data                          # check data/raw/creditcard.csv
    python -m src.data path/to/creditcard.csv   # check another copy
    python -m src.data --download               # fetch the public mirror first

Writes reports/data_checks.json (row and fraud counts, duplicates, SHA-256 of the file).
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
import urllib.request
from pathlib import Path

import pandas as pd

from src.config import load_config, reports_dir, resolve

RAW_FEATURES = ["Time", *[f"V{i}" for i in range(1, 29)], "Amount"]
TARGET = "Class"

# What the published file holds before de-duplication.
EXPECTED_ROWS, EXPECTED_FRAUDS = 284_807, 492
# SHA-256 of the mirror copy that produced the numbers committed in reports/.
# The Kaggle original stores more decimal places, so its hash differs.
MIRROR_SHA256 = "33a178be65174943a07d1736acf5898f6c8e1fa80950dbbe598c17eda968ec59"


def load_raw(path: str | Path) -> pd.DataFrame:
    """Read the CSV and fail loudly if it isn't the dataset the pipeline expects."""
    df = pd.read_csv(path)
    missing = [c for c in RAW_FEATURES + [TARGET] if c not in df.columns]
    if missing:
        raise ValueError(f"missing columns: {missing}")
    df = df[RAW_FEATURES + [TARGET]]
    if df.isna().any().any():
        raise ValueError("dataset has missing values")
    if not df[TARGET].isin([0, 1]).all():
        raise ValueError("Class must be 0 or 1")
    return df.astype({TARGET: int})


def deduplicate(df: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    """Drop exact duplicate rows *before* any split, so no copy can sit on both sides of it.

    Rows with identical features but different labels are counted, not dropped: they are
    contradictory examples rather than copies.
    """
    exact_copy = df.duplicated()
    same_features = df[df.duplicated(RAW_FEATURES, keep=False)]
    conflicts = int((same_features.groupby(RAW_FEATURES)[TARGET].nunique() > 1).sum())
    out = df[~exact_copy].reset_index(drop=True)
    report = {
        "rows_before": len(df),
        "rows_after": len(out),
        "duplicates_dropped": int(exact_copy.sum()),
        "frauds_before": int(df[TARGET].sum()),
        "frauds_after": int(out[TARGET].sum()),
        "fraud_duplicates_dropped": int(df.loc[exact_copy, TARGET].sum()),
        "fraud_rate_after": float(out[TARGET].mean()),
        "same_features_different_label": conflicts,
    }
    return out, report


def load_dataset(path: str | Path) -> tuple[pd.DataFrame, dict]:
    """The one entry point the pipeline uses: validated and de-duplicated data."""
    return deduplicate(load_raw(path))


def sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def data_checks(path: str | Path) -> dict:
    raw = pd.read_csv(path)
    missing_values = int(raw.isna().sum().sum())
    df = load_raw(path)
    _, report = deduplicate(df)
    file_hash = sha256(path)
    return {
        "file": Path(path).name,
        "sha256": file_hash,
        "matches_mirror_used_for_reports": file_hash == MIRROR_SHA256,
        "rows": len(df),
        "frauds": int(df[TARGET].sum()),
        "fraud_rate": float(df[TARGET].mean()),
        "missing_values": missing_values,
        "expected": {"rows": EXPECTED_ROWS, "frauds": EXPECTED_FRAUDS},
        "matches_expected": len(df) == EXPECTED_ROWS and int(df[TARGET].sum()) == EXPECTED_FRAUDS,
        **{k: v for k, v in report.items() if k not in ("rows_before", "frauds_before")},
    }


def download(url: str, target: Path) -> None:
    target.parent.mkdir(parents=True, exist_ok=True)
    partial = target.with_suffix(".part")
    print(f"downloading {url}\n        to {target} ...")
    urllib.request.urlretrieve(url, partial)
    partial.replace(target)  # only a complete file ever gets the real name
    print(f"done ({target.stat().st_size / 1e6:.1f} MB)")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("path", nargs="?", help="CSV to check (default: data.raw_path in the config)")
    parser.add_argument("--download", action="store_true", help="fetch the public mirror first")
    parser.add_argument("--config", default=None)
    args = parser.parse_args(argv)

    cfg = load_config(args.config)
    path = resolve(args.path or cfg["data"]["raw_path"])
    if args.download:
        if path.exists():
            print(f"{path} already exists; delete it to download again")
        else:
            try:
                download(cfg["data"]["download_url"], path)
            except Exception as exc:  # network errors vary by platform
                print(f"download failed: {exc}\nGet creditcard.csv from "
                      "https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud "
                      f"and save it as {path}", file=sys.stderr)
                return 1
    if not path.exists():
        print(f"{path} not found. Run `python -m src.data --download`, or download "
              "creditcard.csv from Kaggle into data/raw/.", file=sys.stderr)
        return 1

    checks = data_checks(path)
    out = reports_dir(cfg) / "data_checks.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(checks, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(checks, indent=2))
    print(f"wrote {out}")
    if not checks["matches_expected"]:
        print(f"WARNING: expected {EXPECTED_ROWS:,} rows and {EXPECTED_FRAUDS} frauds; "
              "find out why before training.", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
