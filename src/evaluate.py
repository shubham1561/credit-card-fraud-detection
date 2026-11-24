"""Metrics on the fraud class, with uncertainty."""
from __future__ import annotations

import numpy as np
from sklearn.metrics import average_precision_score, brier_score_loss, roc_auc_score, roc_curve


def ranking_metrics(y, scores) -> dict:
    return {
        "pr_auc": float(average_precision_score(y, scores)),
        "roc_auc": float(roc_auc_score(y, scores)),
        "no_skill_pr_auc": float(np.mean(y)),  # a random scorer's PR-AUC is the fraud rate
    }


def operating_point(y, scores, threshold: float | None, c_miss: float | None = None,
                    c_review: float = 1.0) -> dict:
    """Counts at one threshold. `None` (or inf) means "flag nothing"."""
    y = np.asarray(y) == 1
    t = np.inf if threshold is None else threshold
    flag = np.asarray(scores) >= t
    tp, fp = int((flag & y).sum()), int((flag & ~y).sum())
    fn, tn = int((~flag & y).sum()), int((~flag & ~y).sum())
    out = {
        "threshold": None if np.isinf(t) else float(t), "alerts": tp + fp,
        "tp": tp, "fp": fp, "fn": fn, "tn": tn,
        "precision": tp / (tp + fp) if tp + fp else 0.0,
        "recall": tp / (tp + fn) if tp + fn else 0.0,
        "fpr": fp / (fp + tn) if fp + tn else 0.0,
    }
    if c_miss is not None:
        out["cost"] = float(c_miss * fn + c_review * fp)
    return out


def recall_at_fpr(y, scores, max_fpr: float = 0.01) -> float:
    """Share of frauds caught while flagging at most `max_fpr` of legitimate transactions."""
    fpr, tpr, _ = roc_curve(y, scores, drop_intermediate=False)
    return float(tpr[fpr <= max_fpr].max())


def precision_at_k(y, scores, k: int) -> dict:
    """If the team can review only k alerts: how many of the k highest scores are fraud."""
    y = np.asarray(y)
    top = np.argsort(-np.asarray(scores), kind="mergesort")[:k]
    frauds = int(y[top].sum())
    return {"k": k, "frauds": frauds, "precision": frauds / k}


# ---------------------------------------------------------------- bootstrap
# Each resample keeps every row's score, so the ranking never changes: only how often each
# row is counted. Sorting once and recomputing weighted cumulative counts per resample gives
# exactly sklearn's average_precision_score (tests/test_core.py checks this) far faster.

def _sort_once(y, scores):
    order = np.argsort(-np.asarray(scores, dtype=float), kind="mergesort")
    s = np.asarray(scores, dtype=float)[order]
    ends = np.r_[np.flatnonzero(np.diff(s)), s.size - 1]  # last row of each tied-score run
    return order, np.asarray(y, dtype=float)[order], ends


def _weighted_ap(sorted_once, weights) -> float:
    order, y_sorted, ends = sorted_once
    w = weights[order]
    tp = np.cumsum(w * y_sorted)[ends]
    fp = np.cumsum(w * (1 - y_sorted))[ends]
    precision = np.divide(tp, tp + fp, out=np.zeros_like(tp), where=(tp + fp) > 0)
    recall = tp / tp[-1]
    return float(np.sum(np.diff(np.r_[0.0, recall]) * precision))  # AP = sum (R_n - R_n-1) P_n


def bootstrap_pr_auc(y, scores_a, scores_b=None, n_boot: int = 2000, seed: int = 0) -> dict:
    """Stratified bootstrap of the evaluation rows.

    Frauds and legitimate rows are resampled separately, so every resample has the same
    fraud count as the real set. Returns the 95% percentile interval for model A's PR-AUC
    and, if B is given, for the paired difference A - B (both scored on the same resampled
    rows) and the share of resamples in which A wins.
    """
    y = np.asarray(y).astype(int)
    rng = np.random.default_rng(seed)
    pos, neg = np.flatnonzero(y == 1), np.flatnonzero(y == 0)
    sorted_a = _sort_once(y, scores_a)
    sorted_b = None if scores_b is None else _sort_once(y, scores_b)
    ap_a, diff = np.empty(n_boot), np.empty(n_boot)
    for b in range(n_boot):
        idx = np.r_[rng.choice(pos, pos.size), rng.choice(neg, neg.size)]
        w = np.bincount(idx, minlength=y.size).astype(float)
        ap_a[b] = _weighted_ap(sorted_a, w)
        if sorted_b is not None:
            diff[b] = ap_a[b] - _weighted_ap(sorted_b, w)
    out = {"pr_auc_ci95": np.percentile(ap_a, [2.5, 97.5]).tolist(), "n_boot": n_boot}
    if sorted_b is not None:
        out["diff"] = float(average_precision_score(y, scores_a) - average_precision_score(y, scores_b))
        out["diff_ci95"] = np.percentile(diff, [2.5, 97.5]).tolist()
        out["share_of_resamples_a_wins"] = float(np.mean(diff > 0))
    return out


# --------------------------------------------------------------- calibration

def expected_vs_actual(y, p, edges=(0.0, 0.01, 0.1, 0.5, 1.0)) -> list[dict]:
    """For calibrated scores, the scores in each band add up to roughly the frauds in it."""
    y, p = np.asarray(y), np.asarray(p)
    rows = []
    for lo, hi in zip(edges[:-1], edges[1:]):
        in_band = (p >= lo) & ((p < hi) if hi < edges[-1] else (p <= hi))
        rows.append({"band": f"{lo:g}-{hi:g}", "n": int(in_band.sum()),
                     "expected_frauds": float(p[in_band].sum()), "actual_frauds": int(y[in_band].sum())})
    return rows


def fit_sigmoid_calibrator(model, X_val, y_val):
    """Platt (sigmoid) calibration fitted on validation rows only, on top of a model that is
    already fitted and stays frozen. Isotonic is avoided: with the few dozen frauds in a
    validation block it overfits."""
    from sklearn.calibration import CalibratedClassifierCV
    from sklearn.frozen import FrozenEstimator

    return CalibratedClassifierCV(FrozenEstimator(model), method="sigmoid").fit(X_val, y_val)


def calibration_report(y_test, raw, calibrated) -> dict:
    """E4: are the scores probabilities? Brier score and expected vs actual frauds per band."""
    return {
        "brier_raw": float(brier_score_loss(y_test, raw)),
        "brier_sigmoid": float(brier_score_loss(y_test, calibrated)),
        "expected_frauds_raw": float(np.sum(raw)),
        "expected_frauds_sigmoid": float(np.sum(calibrated)),
        "actual_frauds": int(np.sum(y_test)),
        "bands_raw": expected_vs_actual(y_test, raw),
        "bands_sigmoid": expected_vs_actual(y_test, calibrated),
    }
