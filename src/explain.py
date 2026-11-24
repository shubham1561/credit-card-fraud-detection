"""Per-transaction explanations from XGBoost's built-in TreeSHAP (no shap package needed)."""
from __future__ import annotations

import pandas as pd
import xgboost as xgb


def tree_shap(model, X: pd.DataFrame) -> pd.DataFrame:
    """SHAP values in log-odds units. Each row's values plus the bias sum to the model's
    margin, so they say how much each feature pushed that transaction's score up or down."""
    contrib = model.get_booster().predict(xgb.DMatrix(X), pred_contribs=True)
    return pd.DataFrame(contrib[:, :-1], index=X.index, columns=X.columns)  # last column = bias


def mean_abs_shap(shap: pd.DataFrame) -> dict[str, float]:
    """Global importance: the average size of each feature's push, largest first."""
    return {k: float(v) for k, v in shap.abs().mean().sort_values(ascending=False).items()}


def top_contributions(shap_row: pd.Series, k: int = 3) -> list[list]:
    """The k features that moved this one score most, with the signed push of each."""
    top = shap_row.reindex(shap_row.abs().sort_values(ascending=False).index[:k])
    return [[name, round(float(value), 3)] for name, value in top.items()]
