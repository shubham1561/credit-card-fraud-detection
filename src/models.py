"""Model factory.

`weighted` switches class weighting on or off with nothing else changed, which is what
experiment E2 needs. Keyword arguments from configs/config.yaml override the defaults.
"""
from __future__ import annotations

import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier

MODEL_NAMES = ("logreg", "rf", "xgb")
LABELS = {"logreg": "Logistic regression", "rf": "Random forest", "xgb": "XGBoost"}


def neg_pos_ratio(y) -> float:
    """Legitimate rows per fraud: the weight each fraud gets under class weighting."""
    y = np.asarray(y)
    return float((y == 0).sum() / (y == 1).sum())


def make_model(name: str, weighted: bool, neg_pos_ratio: float, seed: int = 42, **params):
    """Both weighting schemes give the frauds, together, the same total weight as all the
    legitimate rows together, so the three model families are weighted comparably."""
    if name == "logreg":
        # The scaler lives inside the pipeline, so it is only ever fitted on training rows.
        lr = LogisticRegression(class_weight="balanced" if weighted else None,
                                **({"C": 1.0, "max_iter": 5000} | params))
        return make_pipeline(StandardScaler(), lr)
    if name == "rf":
        return RandomForestClassifier(
            class_weight="balanced" if weighted else None, n_jobs=-1, random_state=seed,
            **({"n_estimators": 300, "min_samples_leaf": 2} | params),
        )
    if name == "xgb":
        defaults = {"n_estimators": 400, "learning_rate": 0.05, "max_depth": 5,
                    "subsample": 0.8, "colsample_bytree": 0.8}
        return XGBClassifier(
            scale_pos_weight=neg_pos_ratio if weighted else 1.0, tree_method="hist",
            n_jobs=-1, random_state=seed, **(defaults | params),
        )
    raise ValueError(f"unknown model {name!r}; choose from {MODEL_NAMES}")
