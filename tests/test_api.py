"""The scoring API, served from a small bundle trained on synthetic data, so these tests
run without the real dataset or a trained model."""
from __future__ import annotations

import joblib
import numpy as np
import pytest

pytest.importorskip("fastapi")
from fastapi.testclient import TestClient  # noqa: E402

from src.data import deduplicate  # noqa: E402
from src.evaluate import fit_sigmoid_calibrator  # noqa: E402
from src.features import MODEL_FEATURES, build_features  # noqa: E402
from src.models import make_model, neg_pos_ratio  # noqa: E402
from tests.conftest import synthetic_frame  # noqa: E402

TX = {"Time": 3600.0, "Amount": 149.62, **{f"V{i}": 0.1 for i in range(1, 29)}}


def make_client(tmp_path, monkeypatch, threshold: float):
    df = deduplicate(synthetic_frame(n=2000))[0]
    X, y = build_features(df), df["Class"].to_numpy()
    model = make_model("xgb", True, neg_pos_ratio(y), n_estimators=20, max_depth=3).fit(X, y)
    bundle = {"model": model, "calibrated_model": fit_sigmoid_calibrator(model, X, y),
              "threshold": threshold, "version": "test-bundle", "features": MODEL_FEATURES}
    path = tmp_path / "bundle.joblib"
    joblib.dump(bundle, path)
    monkeypatch.setenv("MODEL_PATH", str(path))
    from app.main import app

    return TestClient(app)


@pytest.fixture
def client(tmp_path, monkeypatch):
    with make_client(tmp_path, monkeypatch, threshold=0.5) as c:  # `with` runs the startup that loads the bundle
        yield c


def test_health_reports_the_model_version(client):
    assert client.get("/health").json() == {"status": "ok", "model_version": "test-bundle"}


def test_predict_returns_score_probability_and_decision(client):
    body = client.post("/predict", json=TX).json()
    assert 0.0 <= body["fraud_score"] <= 1.0 and 0.0 <= body["fraud_probability"] <= 1.0
    assert body["flag_for_review"] == (body["fraud_score"] >= body["threshold"])


def test_a_fraud_like_transaction_scores_higher(client):
    fraud_like = TX | {"V14": -4.0, "V12": -3.0, "V4": 3.0}  # the synthetic fraud signal
    assert client.post("/predict", json=fraud_like).json()["fraud_score"] > client.post("/predict", json=TX).json()["fraud_score"]


@pytest.mark.parametrize("bad", [
    TX | {"Amount": -5.0},                              # negative amount
    {k: v for k, v in TX.items() if k != "V7"},         # missing feature
    TX | {"Class": 1},                                  # unknown field
    TX | {"V3": "high"},                                # not a number
])
def test_rejects_bad_input(client, bad):
    assert client.post("/predict", json=bad).status_code == 422


def test_rejects_nan(client):
    body = '{"Time": 1.0, "Amount": NaN, ' + ", ".join(f'"V{i}": 0.1' for i in range(1, 29)) + "}"
    response = client.post("/predict", content=body, headers={"content-type": "application/json"})
    assert response.status_code == 422


def test_flag_nothing_bundle_never_flags(tmp_path, monkeypatch):
    with make_client(tmp_path, monkeypatch, threshold=np.inf) as c:
        body = c.post("/predict", json=TX | {"V14": -6.0}).json()
    assert body["flag_for_review"] is False and body["threshold"] is None


# ------------------------------------------------------------------- demo endpoints
@pytest.fixture
def demo_client(tmp_path, monkeypatch):
    csv = tmp_path / "creditcard.csv"
    synthetic_frame(n=3000).to_csv(csv, index=False)
    monkeypatch.setenv("DATA_PATH", str(csv))
    with make_client(tmp_path, monkeypatch, threshold=0.5) as c:
        yield c


def test_demo_test_block_counts_add_up(demo_client):
    body = demo_client.get("/demo/test-block").json()
    assert body["frauds_caught"] + body["frauds_missed"] == body["frauds"] > 0
    assert body["frauds_caught"] + body["false_alerts"] == body["alerts_raised"]


def test_demo_sample_labels_every_decision(demo_client):
    body = demo_client.get("/demo/sample", params={"frauds": 3, "legitimate": 4, "seed": 0}).json()
    assert len(body["transactions"]) == 7 and sum(body["summary"].values()) == 7
    assert {t["actual"] for t in body["transactions"]} == {"fraud", "legitimate"}


def test_demo_transaction_can_be_scored(demo_client):
    tx = demo_client.get("/demo/transaction", params={"kind": "fraud", "seed": 0}).json()
    assert demo_client.post("/predict", json=tx).status_code == 200


def test_demo_without_data_says_how_to_get_it(tmp_path, monkeypatch):
    monkeypatch.setenv("DATA_PATH", str(tmp_path / "missing.csv"))
    with make_client(tmp_path, monkeypatch, threshold=0.5) as c:
        assert c.get("/demo/test-block").status_code == 503


def test_home_page_is_served(client):
    r = client.get("/")
    assert r.status_code == 200 and "Card Fraud Detector" in r.text
