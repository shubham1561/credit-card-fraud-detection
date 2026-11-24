"""Turn scores into a decision: flag a transaction for review when score >= threshold."""
from __future__ import annotations

import numpy as np


def cost_curve(y, scores, c_miss: float, c_review: float = 1.0):
    """Exact total cost of flagging every score >= t, for each distinct score t.

    One sort, then cumulative counts: no grid of thresholds to miss the optimum between.
    Rows with tied scores are flagged together, because a threshold can't separate them.
    The first entry (t = inf) means "flag nothing". Cost = c_miss * FN + c_review * FP.
    Returns (thresholds, cost, tp, fp), thresholds in decreasing order.
    """
    y, scores = np.asarray(y), np.asarray(scores, dtype=float)
    order = np.argsort(-scores, kind="mergesort")
    s, yy = scores[order], y[order]
    last = np.r_[np.flatnonzero(np.diff(s)), s.size - 1]  # last row of each run of tied scores
    tp = np.r_[0, np.cumsum(yy)[last]]
    fp = np.r_[0, np.cumsum(1 - yy)[last]]
    thresholds = np.r_[np.inf, s[last]]
    cost = c_miss * (yy.sum() - tp) + c_review * fp
    return thresholds, cost, tp, fp


def best_threshold(y, scores, c_miss: float, c_review: float = 1.0) -> dict:
    """Lowest-cost threshold. np.argmin takes the first minimum, and thresholds decrease,
    so among equal costs the highest threshold wins: the one that raises the fewest alerts."""
    t, cost, tp, fp = cost_curve(y, scores, c_miss, c_review)
    k = int(np.argmin(cost))
    return {"threshold": float(t[k]), "cost": float(cost[k]), "tp": int(tp[k]), "fp": int(fp[k]),
            "alerts": int(tp[k] + fp[k]), "frauds": int(np.sum(y))}


def threshold_sweep(y, scores, ratios=(2, 5, 10, 20, 50, 100, 200, 500), c_review: float = 1.0) -> list[dict]:
    """How the cost-optimal threshold moves as a miss gets more expensive than a review.

    `calibrated_theory` is where the threshold would sit if scores were calibrated
    probabilities: flag when p * c_miss > (1 - p) * c_review, i.e. p > 1 / (1 + r).
    """
    return [{"cost_ratio": r, **best_threshold(y, scores, c_miss=r * c_review, c_review=c_review),
             "calibrated_theory": 1 / (1 + r)} for r in ratios]
