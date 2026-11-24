"""Shared fixtures: a small synthetic table with the real schema, so the tests never need
the 100 MB dataset."""
from __future__ import annotations

import copy

import numpy as np
import pandas as pd
import pytest
import yaml

from src.config import load_config


def synthetic_frame(n: int = 6000, fraud_rate: float = 0.03, n_duplicates: int = 30, seed: int = 0) -> pd.DataFrame:
    """Same 31 columns as creditcard.csv, a learnable fraud signal and planted duplicates."""
    rng = np.random.default_rng(seed)
    y = (rng.random(n) < fraud_rate).astype(int)
    V = rng.normal(size=(n, 28))
    V[:, 13] -= 2.5 * y  # V14
    V[:, 11] -= 1.5 * y  # V12
    V[:, 3] += 1.5 * y   # V4
    df = pd.DataFrame(V, columns=[f"V{i}" for i in range(1, 29)])
    df.insert(0, "Time", np.sort(rng.uniform(0, 172_800, n)).round())
    df["Amount"] = np.round(rng.lognormal(3, 1.5, n), 2)
    df["Class"] = y
    copies = df.sample(n_duplicates, random_state=seed)
    return pd.concat([df, copies], ignore_index=True)


@pytest.fixture
def small_config(tmp_path):
    """The real config pointed at a synthetic CSV and a temporary output folder, with small
    models and few bootstrap resamples so the whole pipeline runs in seconds."""
    csv = tmp_path / "data" / "creditcard.csv"
    csv.parent.mkdir(parents=True)
    synthetic_frame().to_csv(csv, index=False)

    cfg = copy.deepcopy(load_config())
    cfg["data"]["raw_path"] = str(csv)
    cfg["paths"] = {"reports": str(tmp_path / "reports"), "models": str(tmp_path / "models")}
    cfg["bundle"]["path"] = str(tmp_path / "models" / "model_bundle.joblib")
    cfg["protocols"]["random"]["cv_folds"] = 3
    cfg["models"]["rf"] = {"n_estimators": 20, "min_samples_leaf": 2}
    cfg["models"]["xgb"] = {"n_estimators": 30, "max_depth": 3}
    cfg["evaluation"]["bootstrap_resamples"] = 50

    path = tmp_path / "config.yaml"
    path.write_text(yaml.safe_dump(cfg), encoding="utf-8")
    return cfg, path
