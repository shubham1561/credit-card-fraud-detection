"""Real held-out transactions for demonstrating the API.

The demo uses Protocol B's test block: the latest 20% of the file by time, which the served
model never saw during training or threshold selection. It needs data/raw/creditcard.csv
(or DATA_PATH); without it, the demo endpoints answer 503 and /predict has no examples.
"""
from __future__ import annotations

import os
from functools import lru_cache
from pathlib import Path

import pandas as pd

from src.config import load_config, resolve
from src.data import RAW_FEATURES, TARGET, load_dataset
from src.splits import time_split


def data_path() -> Path:
    return resolve(os.environ.get("DATA_PATH") or load_config()["data"]["raw_path"])


@lru_cache(maxsize=2)
def _test_block(path: str) -> pd.DataFrame | None:
    if not Path(path).exists():
        return None
    df, _ = load_dataset(path)
    fractions = load_config()["protocols"]["time"]["fractions"]
    return time_split(df, fractions)[2].reset_index(drop=True)


def test_block() -> pd.DataFrame | None:
    """Protocol B's test block, loaded once and cached."""
    return _test_block(str(data_path()))


def as_request(row: pd.Series) -> dict:
    """One row in the shape POST /predict expects (the label is left out)."""
    return {c: float(row[c]) for c in RAW_FEATURES}


def openapi_examples() -> dict:
    """Ready-made request bodies for the /docs page: real test-block transactions."""
    block = test_block()
    if block is None:
        return {}
    fraud = block[block[TARGET] == 1].iloc[0]
    legit = block[block[TARGET] == 0].iloc[0]
    return {
        "fraud": {"summary": "A real fraud from the held-out test block",
                  "description": f"Labelled fraud in the data; amount {fraud['Amount']:.2f}.",
                  "value": as_request(fraud)},
        "legitimate": {"summary": "A real legitimate transaction from the held-out test block",
                       "description": f"Labelled legitimate in the data; amount {legit['Amount']:.2f}.",
                       "value": as_request(legit)},
    }
