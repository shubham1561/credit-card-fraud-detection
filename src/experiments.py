"""Experiments that compare decisions rather than models.

E2  class weights vs threshold tuning, from the selection scores train.py already has
    (validation block, or out-of-fold predictions) for the weighted and unweighted
    versions of one model.
E6  amount-dependent costs: a missed fraud costs its own amount, not a fixed price.
"""
from __future__ import annotations

import numpy as np
from scipy.stats import spearmanr

from src.evaluate import bootstrap_pr_auc, ranking_metrics
from src.threshold import best_threshold


def _finite(t: float) -> float | None:
    return None if np.isinf(t) else float(t)


def odds_theory_threshold(t_unweighted: float | None, weight: float) -> float | None:
    """If weighting each fraud by w multiplied the model's odds by exactly w, an unweighted
    threshold t would map to w*t / (1 - t + w*t) on the weighted model's scale."""
    if t_unweighted is None:
        return None
    t = t_unweighted
    return weight * t / (1 - t + weight * t)


def class_weight_experiment(y_sel, s_weighted, s_unweighted, weight: float, cost_ratio: float,
                            c_review: float = 1.0, n_boot: int = 2000, seed: int = 0) -> dict:
    """E2. Do class weights change the ranking (PR-AUC), or only where the threshold sits?"""
    c_miss = cost_ratio * c_review
    best_w = best_threshold(y_sel, s_weighted, c_miss=c_miss, c_review=c_review)
    best_u = best_threshold(y_sel, s_unweighted, c_miss=c_miss, c_review=c_review)
    t_u = _finite(best_u["threshold"])
    boot = bootstrap_pr_auc(y_sel, s_weighted, s_unweighted, n_boot=n_boot, seed=seed)
    return {
        "weight": weight,
        "pr_auc_weighted": ranking_metrics(y_sel, s_weighted)["pr_auc"],
        "pr_auc_unweighted": ranking_metrics(y_sel, s_unweighted)["pr_auc"],
        "pr_auc_diff": boot["diff"],
        "pr_auc_diff_ci95": boot["diff_ci95"],
        "share_of_resamples_weighted_wins": boot["share_of_resamples_a_wins"],
        "threshold_weighted": _finite(best_w["threshold"]),
        "threshold_unweighted": t_u,
        "odds_theory_threshold_weighted": odds_theory_threshold(t_u, weight),
        "at_threshold_weighted": {k: best_w[k] for k in ("alerts", "tp", "fp", "cost")},
        "at_threshold_unweighted": {k: best_u[k] for k in ("alerts", "tp", "fp", "cost")},
        "spearman_rho": float(spearmanr(s_weighted, s_unweighted).statistic),
    }


# ------------------------------------------------------------------------ E6

def money_cost(flag, y, amount, review_cost: float, handling: float) -> dict:
    """Money terms: a missed fraud costs its amount plus handling; every alert costs a review."""
    flag, fraud = np.asarray(flag, dtype=bool), np.asarray(y) == 1
    missed = ~flag & fraud
    lost = float(np.asarray(amount)[missed].sum() + handling * missed.sum())
    reviews = int(flag.sum())
    return {"total": lost + review_cost * reviews, "fraud_lost": lost, "review_spend": review_cost * reviews,
            "alerts": reviews, "frauds_caught": int((flag & fraud).sum()),
            "false_alerts": int((flag & ~fraud).sum()), "frauds_missed": int(missed.sum())}


def best_money_threshold(y, scores, amount, review_cost: float, handling: float) -> float | None:
    """The single score threshold with the lowest money cost on the rows given (exact, one sort)."""
    y, scores = np.asarray(y), np.asarray(scores, dtype=float)
    miss_value = np.where(y == 1, np.asarray(amount) + handling, 0.0)
    order = np.argsort(-scores, kind="mergesort")
    s = scores[order]
    last = np.r_[np.flatnonzero(np.diff(s)), s.size - 1]
    caught_value = np.r_[0.0, np.cumsum(miss_value[order])[last]]
    alerts = np.r_[0, last + 1]
    cost = miss_value.sum() - caught_value + review_cost * alerts
    k = int(np.argmin(cost))  # first minimum = highest threshold among ties
    return None if k == 0 else float(s[last[k - 1]])


def amount_cost_experiment(y_val, s_val, amount_val, y_test, s_test, p_test, amount_test,
                           constant_threshold: float | None, review_cost: float, handling: float) -> dict:
    """E6. Four policies scored in money on the test block. Every choice uses validation only.

    - flag_nothing: the cost of having no model
    - constant_ratio: the main analysis threshold (a miss costs a fixed number of reviews)
    - money_threshold: one threshold chosen on validation by money cost (v1's cost model)
    - per_transaction: flag when calibrated p * (amount + handling) > review cost
    """
    def run(flag):
        return money_cost(flag, y_test, amount_test, review_cost, handling)

    t_money = best_money_threshold(y_val, s_val, amount_val, review_cost, handling)
    s_test, p_test, amount_test = map(np.asarray, (s_test, p_test, amount_test))
    never = np.zeros(s_test.size, dtype=bool)
    return {
        "review_cost": review_cost,
        "miss_handling_cost": handling,
        "money_threshold_from_validation": t_money,
        "policies": {
            "flag_nothing": run(never),
            "constant_ratio": run(never if constant_threshold is None else s_test >= constant_threshold),
            "money_threshold": run(never if t_money is None else s_test >= t_money),
            "per_transaction": run(p_test * (amount_test + handling) > review_cost),
        },
    }
