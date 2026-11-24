"""The two evaluation protocols. Both expect de-duplicated data."""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split


def random_split(df: pd.DataFrame, test_size: float = 0.2, seed: int = 42):
    """Protocol A: stratified hold-out, so both sides keep the same fraud rate."""
    train, test = train_test_split(df, test_size=test_size, stratify=df["Class"], random_state=seed)
    return train, test


def time_split(df: pd.DataFrame, fractions=(0.6, 0.2, 0.2)):
    """Protocol B: earliest rows train, the next block validates, the latest block tests.

    Each cut point moves forward past tied timestamps, so no second of the file appears in
    two blocks and every validation and test row is strictly later than all training rows.
    """
    if len(fractions) != 3 or not np.isclose(sum(fractions), 1.0):
        raise ValueError("fractions must be three numbers that sum to 1")
    d = df.sort_values("Time", kind="mergesort").reset_index(drop=True)
    t = d["Time"].to_numpy()
    cuts = [min(max(int(round(len(d) * f)), 1), len(d)) for f in np.cumsum(fractions[:2])]
    i, j = (int(np.searchsorted(t, t[c - 1], side="right")) for c in cuts)
    return d.iloc[:i], d.iloc[i:j], d.iloc[j:]


def describe(part: pd.DataFrame) -> dict:
    """Size, fraud count and time span of one block."""
    return {
        "rows": len(part),
        "frauds": int(part["Class"].sum()),
        "fraud_rate": float(part["Class"].mean()),
        "hours": [round(float(part["Time"].min()) / 3600, 2), round(float(part["Time"].max()) / 3600, 2)],
    }
