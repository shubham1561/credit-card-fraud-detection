"""End-to-end: both protocols, the report and the data checks on a small synthetic table."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from src import data, report, train
from src.models import MODEL_NAMES
from src.threshold import best_threshold


def test_both_protocols_and_the_report_run_end_to_end(small_config):
    cfg, config_path = small_config
    for protocol in ("random", "time"):
        out = train.run(protocol, cfg)
        assert set(out["models"]) == {f"{n}_{w}" for n in MODEL_NAMES for w in "wu"}

    rep = Path(cfg["paths"]["reports"])
    time_metrics = json.loads((rep / "metrics" / "time.json").read_text(encoding="utf-8"))  # strict JSON: no NaN or inf
    split = time_metrics["split"]
    assert split["train"]["hours"][1] <= split["val"]["hours"][0] <= split["test"]["hours"][0]
    for res in time_metrics["models"].values():
        lo, hi = res["test"]["pr_auc_ci95"]
        assert 0 <= lo <= hi <= 1
    assert Path(cfg["bundle"]["path"]).exists()

    assert report.main(["--config", str(config_path)]) == 0
    assert (rep / "results.md").read_text(encoding="utf-8").startswith("# Results")
    assert len(list((rep / "figures").glob("*.png"))) == 9


def test_thresholds_come_from_selection_scores_not_the_test_set(small_config):
    cfg, _ = small_config
    out = train.run("time", cfg)
    scores = Path(cfg["paths"]["reports"]) / "scores"
    y_sel = np.load(scores / "time_y_sel.npy")
    r = cfg["costs"]["main_cost_ratio"]
    for k, res in out["models"].items():
        s_sel = np.load(scores / f"time_{k}_sel.npy")
        assert res["threshold"]["threshold"] == best_threshold(y_sel, s_sel, c_miss=r)["threshold"]
        assert len(s_sel) == out["split"]["val"]["rows"]  # selection = validation block


def test_data_checks_report_a_file_that_is_not_the_real_dataset(small_config):
    cfg, config_path = small_config
    assert data.main([cfg["data"]["raw_path"], "--config", str(config_path)]) == 1  # synthetic: counts differ
    checks = json.loads((Path(cfg["paths"]["reports"]) / "data_checks.json").read_text(encoding="utf-8"))
    assert checks["duplicates_dropped"] == 30 and checks["matches_expected"] is False
