"""Unit tests for the pieces every result depends on."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from sklearn.metrics import average_precision_score

from src.data import RAW_FEATURES, deduplicate
from src.evaluate import (_sort_once, _weighted_ap, bootstrap_pr_auc, operating_point,
                          precision_at_k, recall_at_fpr)
from src.experiments import best_money_threshold, money_cost, odds_theory_threshold
from src.features import MODEL_FEATURES, build_features
from src.models import make_model
from src.splits import random_split, time_split
from src.threshold import best_threshold, cost_curve, threshold_sweep
from tests.conftest import synthetic_frame


# ------------------------------------------------------------------ data
def test_deduplicate_counts():
    row = {c: 0.0 for c in RAW_FEATURES} | {"Class": 0}
    df = pd.DataFrame([row, row, row | {"Amount": 5.0}])
    out, report = deduplicate(df)
    assert len(out) == 2 and report["duplicates_dropped"] == 1


def test_deduplicate_counts_conflicting_labels_without_dropping_them():
    row = {c: 1.0 for c in RAW_FEATURES}
    df = pd.DataFrame([row | {"Class": 0}, row | {"Class": 1}])
    out, report = deduplicate(df)
    assert len(out) == 2 and report["same_features_different_label"] == 1


# -------------------------------------------------------------- features
def test_features_drop_raw_time_and_encode_hour():
    df = pd.DataFrame({f"V{i}": [0.0, 0.0] for i in range(1, 29)})
    df["Time"] = [0.0, 86400 + 6 * 3600.0]  # hour 6 of day 2
    df["Amount"] = [0.0, 99.0]
    X = build_features(df)
    assert list(X.columns) == MODEL_FEATURES and "Time" not in X
    assert np.allclose(X["hour_sin"] ** 2 + X["hour_cos"] ** 2, 1)
    assert np.isclose(X.loc[1, "hour_sin"], 1.0)
    assert np.isclose(X.loc[1, "log_amount"], np.log(100.0))


# ---------------------------------------------------------------- splits
def test_time_blocks_do_not_overlap():
    rng = np.random.default_rng(0)
    df = pd.DataFrame({"Time": np.sort(rng.integers(0, 1000, 5000)).astype(float),
                       "Class": rng.integers(0, 2, 5000)})  # plenty of tied timestamps
    train, val, test = time_split(df)
    assert len(train) + len(val) + len(test) == len(df)
    assert train["Time"].max() < val["Time"].min()
    assert val["Time"].max() < test["Time"].min()


def test_random_split_keeps_the_fraud_rate():
    df = deduplicate(synthetic_frame())[0]
    train, test = random_split(df, test_size=0.2, seed=42)
    assert abs(train["Class"].mean() - test["Class"].mean()) < 0.002
    assert set(train.index).isdisjoint(test.index)


# ------------------------------------------------------------- threshold
def test_cost_curve_matches_brute_force():
    rng = np.random.default_rng(1)
    y = rng.integers(0, 2, 300)
    s = rng.integers(0, 50, 300) / 50  # plenty of tied scores
    thresholds, cost, _, _ = cost_curve(y, s, c_miss=7, c_review=1)
    for t, c in zip(thresholds, cost):
        flag = s >= t
        assert c == 7 * (~flag & (y == 1)).sum() + (flag & (y == 0)).sum()


def test_flags_nothing_when_a_review_costs_more_than_a_miss():
    y, s = np.array([0, 1]), np.array([0.9, 0.1])
    assert best_threshold(y, s, c_miss=1, c_review=100)["threshold"] == np.inf


def test_equal_costs_go_to_the_highest_threshold():
    # flagging the top row alone (one miss) or all three rows (one false alert) both cost 1
    y, s = np.array([1, 0, 1]), np.array([0.9, 0.8, 0.1])
    assert best_threshold(y, s, c_miss=1, c_review=1)["threshold"] == 0.9


def test_sweep_moves_threshold_down_as_misses_get_dearer():
    rng = np.random.default_rng(2)
    y = rng.integers(0, 2, 500)
    s = np.clip(0.3 * y + rng.random(500) * 0.7, 0, 1)
    t = [row["threshold"] for row in threshold_sweep(y, s, ratios=(1, 5, 50))]
    assert t[0] >= t[1] >= t[2]


# -------------------------------------------------------------- evaluate
def test_fast_weighted_ap_matches_sklearn():
    rng = np.random.default_rng(3)
    y = rng.integers(0, 2, 2000)
    s = np.round(rng.random(2000), 2)  # ties included
    w = rng.integers(0, 4, 2000).astype(float)  # zero weights included, as in a bootstrap
    expected = average_precision_score(y[w > 0], s[w > 0], sample_weight=w[w > 0])
    assert np.isclose(_weighted_ap(_sort_once(y, s), w), expected)
    assert np.isclose(_weighted_ap(_sort_once(y, s), np.ones(2000)), average_precision_score(y, s))


def test_bootstrap_interval_contains_estimate_and_identical_models_tie():
    rng = np.random.default_rng(4)
    y = (rng.random(3000) < 0.05).astype(int)
    s = y * 0.5 + rng.random(3000)
    res = bootstrap_pr_auc(y, s, s, n_boot=200, seed=0)
    lo, hi = res["pr_auc_ci95"]
    assert lo <= average_precision_score(y, s) <= hi
    assert res["diff_ci95"] == [0.0, 0.0] and res["share_of_resamples_a_wins"] == 0.0


def test_operating_point_counts():
    y, s = np.array([1, 1, 0, 0, 0]), np.array([0.9, 0.2, 0.8, 0.1, 0.05])
    op = operating_point(y, s, 0.5, c_miss=10)
    assert (op["tp"], op["fp"], op["fn"], op["tn"]) == (1, 1, 1, 2)
    assert op["cost"] == 10 * 1 + 1
    assert operating_point(y, s, None)["alerts"] == 0


def test_recall_at_fpr_and_precision_at_k():
    y = np.array([1, 0, 1, 0, 0, 0, 0, 0, 0, 0])
    s = np.array([0.95, 0.9, 0.85, 0.5, 0.4, 0.3, 0.2, 0.1, 0.05, 0.01])
    assert recall_at_fpr(y, s, max_fpr=0.0) == 0.5
    assert recall_at_fpr(y, s, max_fpr=0.125) == 1.0
    assert precision_at_k(y, s, 3) == {"k": 3, "frauds": 2, "precision": 2 / 3}


# ----------------------------------------------------------- experiments
def test_odds_theory_threshold():
    assert np.isclose(odds_theory_threshold(0.5, weight=1.0), 0.5)
    assert np.isclose(odds_theory_threshold(0.1, weight=9.0), 0.5)  # odds 1/9 -> 1
    assert odds_theory_threshold(None, weight=9.0) is None


def test_money_threshold_matches_brute_force():
    rng = np.random.default_rng(5)
    y = (rng.random(400) < 0.1).astype(int)
    s = np.round(rng.random(400), 2)
    amount = np.round(rng.lognormal(3, 1, 400), 2)
    t = best_money_threshold(y, s, amount, review_cost=5.0, handling=2.0)
    candidates = [np.inf, *np.unique(s)[::-1]]  # highest first, so ties resolve the same way
    costs = [money_cost(s >= c, y, amount, 5.0, 2.0)["total"] for c in candidates]
    best = candidates[int(np.argmin(costs))]
    assert (t is None and best == np.inf) or np.isclose(t, best)


# ---------------------------------------------------------------- models
@pytest.mark.parametrize("name", ["logreg", "rf", "xgb"])
def test_weighting_flag_changes_only_the_weights(name):
    weighted, plain = make_model(name, True, 600.0), make_model(name, False, 600.0)
    if name == "xgb":
        assert weighted.get_params()["scale_pos_weight"] == 600.0
        assert plain.get_params()["scale_pos_weight"] == 1.0
    else:
        est_w = weighted[-1] if name == "logreg" else weighted
        est_p = plain[-1] if name == "logreg" else plain
        assert est_w.get_params()["class_weight"] == "balanced"
        assert est_p.get_params()["class_weight"] is None
