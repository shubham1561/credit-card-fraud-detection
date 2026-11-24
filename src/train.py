"""Train and evaluate every model under one evaluation protocol.

    python -m src.train --protocol random    # Protocol A: stratified 80/20 split + 5-fold CV
    python -m src.train --protocol time      # Protocol B: time-ordered 60/20/20 split

Writes reports/metrics/<protocol>.json and the scores behind it in reports/scores/.
The time protocol also saves the model bundle the API serves (models/model_bundle.joblib).

The order of operations is the leakage control:
  1. de-duplicate, then split;
  2. choose each threshold on selection scores only (out-of-fold for A, validation for B);
  3. score each test set once, after every choice is frozen.
"""
from __future__ import annotations

import argparse
import importlib
import json
import platform
import subprocess
import sys
import time
from pathlib import Path

import joblib
import numpy as np
from sklearn.metrics import average_precision_score
from sklearn.model_selection import StratifiedKFold

from src.config import load_config, reports_dir, resolve
from src.data import TARGET, load_dataset
from src.evaluate import (bootstrap_pr_auc, calibration_report, fit_sigmoid_calibrator,
                          operating_point, precision_at_k, ranking_metrics, recall_at_fpr)
from src.experiments import amount_cost_experiment, class_weight_experiment
from src.explain import mean_abs_shap, top_contributions, tree_shap
from src.features import MODEL_FEATURES, build_features, hour_of_day
from src.models import MODEL_NAMES, make_model, neg_pos_ratio
from src.splits import describe, random_split, time_split
from src.threshold import best_threshold, threshold_sweep

# Recorded in every metrics JSON. Read from the imported modules, not from pip metadata, so
# installs under another distribution name (xgboost-cpu, conda builds) are reported too.
PACKAGES = {"numpy": "numpy", "pandas": "pandas", "scipy": "scipy", "scikit-learn": "sklearn",
            "xgboost": "xgboost", "matplotlib": "matplotlib", "pyyaml": "yaml", "joblib": "joblib"}


def key(name: str, weighted: bool) -> str:
    return f"{name}_{'w' if weighted else 'u'}"


def log(msg: str) -> None:
    print(msg, flush=True)


def out_of_fold_scores(name, weighted, params, X, y, folds: int, seed: int):
    """Protocol A selection scores: each training row is scored by a model that never saw it."""
    oof = np.zeros(len(y))
    fold_pr_auc = []
    for tr, va in StratifiedKFold(n_splits=folds, shuffle=True, random_state=seed).split(X, y):
        model = make_model(name, weighted, neg_pos_ratio(y[tr]), seed, **params)
        model.fit(X.iloc[tr], y[tr])
        oof[va] = model.predict_proba(X.iloc[va])[:, 1]
        fold_pr_auc.append(float(average_precision_score(y[va], oof[va])))
    return oof, fold_pr_auc


def git_commit() -> str | None:
    try:
        out = subprocess.run(["git", "rev-parse", "--short", "HEAD"], capture_output=True, text=True,
                             cwd=Path(__file__).resolve().parents[1], timeout=10)
        return out.stdout.strip() or None
    except (OSError, subprocess.SubprocessError):
        return None


def environment() -> dict:
    versions = {}
    for label, module in PACKAGES.items():
        try:
            versions[label] = importlib.import_module(module).__version__
        except (ImportError, AttributeError):  # a missing version shouldn't stop a training run
            versions[label] = None
    return {"python": platform.python_version(), "platform": platform.platform(), **versions}


def to_json(obj):
    """JSON has no infinity: an infinite threshold ("flag nothing") is stored as null."""
    if isinstance(obj, dict):
        return {str(k): to_json(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [to_json(v) for v in obj]
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (float, np.floating)):
        return None if np.isinf(obj) else float(obj)
    if isinstance(obj, np.bool_):
        return bool(obj)
    return obj


def run(protocol: str, cfg: dict) -> dict:
    started = time.time()
    seed = cfg["seed"]
    costs, ev = cfg["costs"], cfg["evaluation"]
    ratio_main, c_review = costs["main_cost_ratio"], costs["c_review"]
    c_miss = ratio_main * c_review
    n_boot, boot_seed = ev["bootstrap_resamples"], ev["bootstrap_seed"]
    score_dir = reports_dir(cfg) / "scores"
    score_dir.mkdir(parents=True, exist_ok=True)

    # 1. data: validate and de-duplicate BEFORE splitting
    df, data_report = load_dataset(resolve(cfg["data"]["raw_path"]))
    if protocol == "random":
        settings = cfg["protocols"]["random"]
        train, test = random_split(df, settings["test_size"], seed)
        parts = {"train": train, "test": test}
    elif protocol == "time":
        train, val, test = time_split(df, cfg["protocols"]["time"]["fractions"])
        parts = {"train": train, "val": val, "test": test}
    else:
        raise ValueError(f"unknown protocol {protocol!r}")
    split = {name: describe(part) for name, part in parts.items()}
    log(f"[{protocol}] " + "  ".join(f"{n}: {s['rows']:,} rows / {s['frauds']} frauds" for n, s in split.items()))

    X = {name: build_features(part) for name, part in parts.items()}
    y = {name: part[TARGET].to_numpy() for name, part in parts.items()}
    weight = neg_pos_ratio(y["train"])  # legitimate rows per fraud in the training data
    y_sel = y["train"] if protocol == "random" else y["val"]
    np.save(score_dir / f"{protocol}_y_sel.npy", y_sel)
    np.save(score_dir / f"{protocol}_y_test.npy", y["test"])

    # 2. every model, with and without class weights
    results, sel_scores, test_scores, fitted = {}, {}, {}, {}
    for name in MODEL_NAMES:
        params = (cfg.get("models") or {}).get(name) or {}
        for weighted in cfg["class_weighting"]:
            k, tick = key(name, weighted), time.time()
            if protocol == "random":
                s_sel, folds = out_of_fold_scores(name, weighted, params, X["train"], y["train"],
                                                  settings["cv_folds"], seed)
                model = make_model(name, weighted, weight, seed, **params).fit(X["train"], y["train"])
            else:
                model = make_model(name, weighted, weight, seed, **params).fit(X["train"], y["train"])
                s_sel, folds = model.predict_proba(X["val"])[:, 1], None
            s_test = model.predict_proba(X["test"])[:, 1]  # the only time the test set is scored
            fitted[k], sel_scores[k], test_scores[k] = model, s_sel, s_test
            np.save(score_dir / f"{protocol}_{k}_sel.npy", s_sel)
            np.save(score_dir / f"{protocol}_{k}_test.npy", s_test)

            chosen = best_threshold(y_sel, s_sel, c_miss=c_miss, c_review=c_review)
            selection = ranking_metrics(y_sel, s_sel)
            if folds:
                selection |= {"cv_fold_pr_auc": folds, "cv_mean": float(np.mean(folds)),
                              "cv_sd": float(np.std(folds, ddof=1))}
            test_metrics = ranking_metrics(y["test"], s_test)
            test_metrics |= {
                "pr_auc_ci95": bootstrap_pr_auc(y["test"], s_test, n_boot=n_boot, seed=boot_seed)["pr_auc_ci95"],
                "recall_at_1pct_fpr": recall_at_fpr(y["test"], s_test, ev["recall_at_fpr"]),
                "operating_point": operating_point(y["test"], s_test, chosen["threshold"], c_miss, c_review),
                "top_k": [precision_at_k(y["test"], s_test, kk) for kk in ev["top_k"]],
            }
            results[k] = {
                "model": name, "weighted": weighted, "selection": selection, "threshold": chosen,
                "sweep": threshold_sweep(y_sel, s_sel, costs["sweep_ratios"], c_review),
                "test": test_metrics, "seconds": round(time.time() - tick, 1),
            }
            op = test_metrics["operating_point"]
            log(f"[{protocol}] {k:9s} selection PR-AUC {selection['pr_auc']:.3f} | test PR-AUC "
                f"{test_metrics['pr_auc']:.3f} ROC-AUC {test_metrics['roc_auc']:.3f} | threshold "
                f"{chosen['threshold']:.4g}: {op['tp']}/{op['tp'] + op['fn']} frauds, {op['fp']} false alerts "
                f"({results[k]['seconds']:.0f}s)")

    # 3. paired comparisons on the test set (same resampled rows for both models)
    comparisons = {}
    for a, b in [("xgb_w", "rf_w"), ("xgb_w", "logreg_w"), ("rf_w", "logreg_w")]:
        if a in test_scores and b in test_scores:
            comparisons[f"{a}_minus_{b}"] = bootstrap_pr_auc(y["test"], test_scores[a], test_scores[b],
                                                             n_boot=n_boot, seed=boot_seed)

    # 4. E2: class weights vs threshold, on the selection scores
    e2 = {}
    for name in MODEL_NAMES:
        kw, ku = key(name, True), key(name, False)
        if kw in sel_scores and ku in sel_scores:
            e2[name] = class_weight_experiment(y_sel, sel_scores[kw], sel_scores[ku], weight, ratio_main,
                                               c_review, n_boot=n_boot, seed=boot_seed)

    out = {
        "protocol": protocol, "git_commit": git_commit(), "environment": environment(), "config": cfg,
        "data": data_report, "split": split, "class_weight": weight, "main_cost_ratio": ratio_main,
        "models": results, "comparisons": comparisons, "class_weight_experiment": e2,
    }

    if protocol == "time":
        out |= time_protocol_extras(cfg, parts, X, y, fitted, sel_scores, test_scores, results)

    out["runtime_seconds"] = round(time.time() - started, 1)
    path = reports_dir(cfg) / "metrics" / f"{protocol}.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(to_json(out), indent=2, allow_nan=False) + "\n", encoding="utf-8")
    log(f"[{protocol}] wrote {path} in {out['runtime_seconds']:.0f}s")
    return out


def time_protocol_extras(cfg, parts, X, y, fitted, sel_scores, test_scores, results) -> dict:
    """E4 calibration, E5 explanations, E6 amount-based costs, and the API's model bundle."""
    bundle_cfg = cfg["bundle"]
    amounts = {name: part["Amount"].to_numpy() for name, part in parts.items()}

    # E4: are the scores probabilities? Sigmoid calibration fitted on the validation block.
    calibrators, calibration = {}, {}
    for k in ("xgb_w", "xgb_u"):
        if k in fitted:
            calibrators[k] = fit_sigmoid_calibrator(fitted[k], X["val"], y["val"])
            p_cal = calibrators[k].predict_proba(X["test"])[:, 1]
            calibration[k] = calibration_report(y["test"], test_scores[k], p_cal)
            np.save(reports_dir(cfg) / "scores" / f"time_{k}_test_calibrated.npy", p_cal)

    # E5: what drives the scores, and what the model misses at its chosen threshold
    explanations = {}
    xk = key("xgb", bundle_cfg["weighted"])
    if xk in fitted:
        shap = tree_shap(fitted[xk], X["test"])
        s, yt = test_scores[xk], y["test"]
        t = results[xk]["threshold"]["threshold"]
        hours = hour_of_day(parts["test"]["Time"]).to_numpy()

        def describe_rows(mask, limit=None, by_score=False):
            idx = np.flatnonzero(mask)
            if by_score:
                idx = idx[np.argsort(-s[idx], kind="mergesort")]
            rows = []
            for i in idx[:limit]:
                rows.append({"amount": float(amounts["test"][i]), "hour": int(hours[i]), "score": float(s[i]),
                             "top_shap": top_contributions(shap.iloc[i])})
            return rows

        missed, caught = (yt == 1) & (s < t), (yt == 1) & (s >= t)
        explanations = {
            "model": xk,
            "shap_mean_abs": mean_abs_shap(shap),
            "missed_frauds": describe_rows(missed, by_score=True),
            "caught_frauds_summary": {
                "n": int(caught.sum()), "median_amount": float(np.median(amounts["test"][caught])) if caught.any() else None,
                "median_score": float(np.median(s[caught])) if caught.any() else None},
            "missed_frauds_summary": {
                "n": int(missed.sum()), "median_amount": float(np.median(amounts["test"][missed])) if missed.any() else None,
                "median_score": float(np.median(s[missed])) if missed.any() else None},
            "top_false_alerts": describe_rows((yt == 0) & (s >= t), limit=5, by_score=True),
        }

    # E6: a miss costs its own amount; uses the bundle model's raw and calibrated scores
    bk = key(bundle_cfg["model"], bundle_cfg["weighted"])
    if bk not in calibrators:
        calibrators[bk] = fit_sigmoid_calibrator(fitted[bk], X["val"], y["val"])
    amount_cfg = cfg["costs"]["amount_based"]
    e6 = amount_cost_experiment(
        y["val"], sel_scores[bk], amounts["val"], y["test"], test_scores[bk],
        calibrators[bk].predict_proba(X["test"])[:, 1], amounts["test"],
        constant_threshold=results[bk]["threshold"]["threshold"],
        review_cost=amount_cfg["review_cost"], handling=amount_cfg["miss_handling_cost"])
    e6["model"] = bk

    # the bundle the API serves: model, calibrator and threshold travel together
    threshold = results[bk]["threshold"]["threshold"]
    bundle = {"model": fitted[bk], "calibrated_model": calibrators[bk], "threshold": threshold,
              "version": f"{bk}-time-v2", "features": MODEL_FEATURES,
              "cost_ratio": cfg["costs"]["main_cost_ratio"]}
    path = resolve(bundle_cfg["path"])
    path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(bundle, path)
    log(f"[time] saved model bundle {bundle['version']} (threshold {threshold:.4g}) to {path}")

    return {"calibration": calibration, "explanations": explanations, "amount_based_costs": e6,
            "bundle": {"version": bundle["version"], "threshold": threshold, "path": str(bundle_cfg["path"])}}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--protocol", choices=["random", "time"], required=True)
    parser.add_argument("--config", default=None, help="default: configs/config.yaml")
    args = parser.parse_args(argv)
    cfg = load_config(args.config)
    if not resolve(cfg["data"]["raw_path"]).exists():
        print(f"{cfg['data']['raw_path']} not found: run `python -m src.data --download` first", file=sys.stderr)
        return 1
    run(args.protocol, cfg)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
