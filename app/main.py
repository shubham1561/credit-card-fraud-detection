"""Scoring API.

    uvicorn app.main:app --reload        # demo page at http://127.0.0.1:8000, API docs at /docs

Loads the bundle saved by `python -m src.train --protocol time` from MODEL_PATH
(default models/model_bundle.joblib) and applies the same `build_features` as training.

Demo, all from the /docs page:
  POST /predict          pick "A real fraud ..." from the Examples list, then Execute
  GET  /demo/sample      score a random mix of real held-out frauds and legitimate transactions
  GET  /demo/test-block  score the whole held-out test block: frauds caught, missed, false alerts
"""
from __future__ import annotations

import os
from contextlib import asynccontextmanager

import joblib
import numpy as np
import pandas as pd
from fastapi import Body, FastAPI, HTTPException, Query, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import HTMLResponse, JSONResponse
from pathlib import Path
from pydantic import ConfigDict, Field, create_model

from app.demo import as_request, openapi_examples, test_block
from src.config import resolve
from src.data import TARGET
from src.features import MODEL_FEATURES, build_features

DEFAULT_MODEL_PATH = "models/model_bundle.joblib"


@asynccontextmanager
async def lifespan(app: FastAPI):
    path = resolve(os.environ.get("MODEL_PATH", DEFAULT_MODEL_PATH))
    if not path.exists():
        raise RuntimeError(f"no model bundle at {path}: run `python -m src.train --protocol time` first")
    app.state.bundle = joblib.load(path)
    yield


app = FastAPI(title="Card fraud scoring", version="2.0", lifespan=lifespan)


@app.exception_handler(RequestValidationError)
async def validation_error(request: Request, exc: RequestValidationError):
    """422 with where and why, without echoing the input back: FastAPI's default handler
    repeats the offending value, and a NaN can't be written as JSON, so it would crash."""
    errors = [{"loc": list(e["loc"]), "msg": e["msg"], "type": e["type"]} for e in exc.errors()]
    return JSONResponse(status_code=422, content={"detail": errors})

# Time and Amount must be non-negative; all of V1-V28 are required. Unknown fields, NaN and
# infinity are rejected, so a malformed request gets HTTP 422 instead of a silent score.
Transaction = create_model(
    "Transaction",
    __config__=ConfigDict(extra="forbid", allow_inf_nan=False),
    Time=(float, Field(ge=0, description="seconds since the first transaction in the file")),
    Amount=(float, Field(ge=0)),
    **{f"V{i}": (float, ...) for i in range(1, 29)},
)


PAGE = Path(__file__).parent / "static" / "index.html"


@app.get("/", response_class=HTMLResponse, include_in_schema=False)
def home():
    """The demo page: pick a real transaction and see the model's decision."""
    return PAGE.read_text(encoding="utf-8")


@app.get("/health")
def health(request: Request):
    return {"status": "ok", "model_version": request.app.state.bundle["version"]}


def score_frame(bundle: dict, raw: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    """Raw scores and review flags for rows with the raw columns."""
    X = build_features(raw)[MODEL_FEATURES]
    scores = bundle["model"].predict_proba(X)[:, 1]
    threshold = bundle["threshold"]
    flags = scores >= threshold if np.isfinite(threshold) else np.zeros(len(scores), dtype=bool)
    return scores, flags


EXAMPLES = openapi_examples()  # real test-block transactions for the /docs page, if the data is present


@app.post("/predict")
def predict(request: Request, tx: Transaction = Body(openapi_examples=EXAMPLES or None)):
    """Score one transaction. On /docs, choose an example from the Examples list to try a real one."""
    bundle = request.app.state.bundle
    raw = pd.DataFrame([tx.model_dump()])
    scores, flags = score_frame(bundle, raw)
    threshold = bundle["threshold"]
    response = {
        "decision": "FLAG FOR REVIEW: likely fraud" if flags[0] else "approve: looks legitimate",
        "fraud_score": float(scores[0]),      # the model's raw score, compared with the threshold
        "flag_for_review": bool(flags[0]),
        "threshold": float(threshold) if np.isfinite(threshold) else None,
        "model_version": bundle["version"],
    }
    if bundle.get("calibrated_model") is not None:  # sigmoid-calibrated estimate of P(fraud)
        X = build_features(raw)[MODEL_FEATURES]
        response["fraud_probability"] = float(bundle["calibrated_model"].predict_proba(X)[0, 1])
    return response


def _threshold(request: Request) -> float | None:
    t = request.app.state.bundle["threshold"]
    return float(t) if np.isfinite(t) else None


def _block_or_503() -> pd.DataFrame:
    block = test_block()
    if block is None:
        raise HTTPException(503, "data/raw/creditcard.csv not found: run `python -m src.data --download`")
    return block


VERDICT = {(1, True): "fraud caught", (1, False): "fraud missed",
           (0, True): "false alert", (0, False): "correctly approved"}


@app.get("/demo/sample")
def demo_sample(request: Request, frauds: int = Query(5, ge=0, le=500), legitimate: int = Query(5, ge=0, le=1000),
                seed: int | None = Query(None, description="set it to get the same sample again")):
    """Score a random mix of real transactions from the held-out test block, which the model never
    saw, and compare each decision with the true label. Misses are shown as well as catches."""
    block = _block_or_503()
    rows = pd.concat([block[block[TARGET] == 1].sample(min(frauds, int(block[TARGET].sum())), random_state=seed),
                      block[block[TARGET] == 0].sample(legitimate, random_state=seed)])
    scores, flags = score_frame(request.app.state.bundle, rows)
    results = [{"actual": "fraud" if y else "legitimate", "amount": float(a), "fraud_score": round(float(s), 5),
                "flagged": bool(f), "verdict": VERDICT[(int(y), bool(f))]}
               for y, a, s, f in zip(rows[TARGET], rows["Amount"], scores, flags)]
    results.sort(key=lambda r: -r["fraud_score"])
    counts = pd.Series([r["verdict"] for r in results]).value_counts().to_dict()
    return {"threshold": _threshold(request), "summary": counts, "transactions": results}


@app.get("/demo/test-block")
def demo_test_block(request: Request):
    """Score every transaction in the held-out test block: the numbers in the README, live."""
    block = _block_or_503()
    scores, flags = score_frame(request.app.state.bundle, block)
    fraud = block[TARGET].to_numpy() == 1
    caught, alerts = int((flags & fraud).sum()), int(flags.sum())
    return {"transactions": len(block), "frauds": int(fraud.sum()), "frauds_caught": caught,
            "frauds_missed": int(fraud.sum()) - caught, "false_alerts": alerts - caught,
            "alerts_raised": alerts, "precision": round(caught / alerts, 3) if alerts else None,
            "recall": round(caught / int(fraud.sum()), 3), "threshold": _threshold(request)}


@app.get("/demo/transaction")
def demo_transaction(kind: str = Query("fraud", pattern="^(fraud|legitimate)$"), seed: int | None = None):
    """A random real transaction from the held-out test block, ready to paste into POST /predict."""
    block = _block_or_503()
    row = block[block[TARGET] == (1 if kind == "fraud" else 0)].sample(1, random_state=seed).iloc[0]
    return as_request(row)
