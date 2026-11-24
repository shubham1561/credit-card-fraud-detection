"""Raw columns -> model inputs.

`build_features` is the only feature code in the project. Training and the API both call
it, so the model never sees features built two different ways.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

V_COLS = [f"V{i}" for i in range(1, 29)]
MODEL_FEATURES = V_COLS + ["log_amount", "hour_sin", "hour_cos"]


def hour_of_day(time_seconds: pd.Series) -> pd.Series:
    """Hours since the first transaction in the file, folded onto a 24-hour cycle.

    `Time` is relative to the start of the file, so this is not a known clock hour.
    """
    return (time_seconds // 3600 % 24).astype(int)


def build_features(df: pd.DataFrame) -> pd.DataFrame:
    hour = hour_of_day(df["Time"])
    out = df[V_COLS].copy()
    out["log_amount"] = np.log1p(df["Amount"])  # amounts are heavily right-skewed
    out["hour_sin"] = np.sin(2 * np.pi * hour / 24)  # sine + cosine: hour 23 sits next to hour 0
    out["hour_cos"] = np.cos(2 * np.pi * hour / 24)
    return out[MODEL_FEATURES]  # raw Time is left out on purpose (docs section 5)
