"""Tests for the FastAPI serving layer (serving.py).

Artifacts are tiny and injected — no dependency on a trained pipeline run.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest
from fastapi.testclient import TestClient

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from quenchguard.serving import Artifacts, create_app

SEQ_LEN, N_FEATURES = 4, 3


@pytest.fixture()
def artifacts() -> Artifacts:
    """Tiny LSTM + scaler + HistGBDT wired together."""
    import joblib
    from sklearn.ensemble import HistGradientBoostingClassifier
    from sklearn.preprocessing import StandardScaler

    from quenchguard.lstm import LSTMDetector

    rng = np.random.default_rng(0)
    X = rng.normal(size=(300, N_FEATURES))
    y = (X[:, 0] + 0.4 * X[:, 1] > 0).astype(int)

    scaler = StandardScaler().fit(X)
    gbdt = HistGradientBoostingClassifier(max_iter=20, random_state=0)
    gbdt.fit(scaler.transform(X), y)

    model = LSTMDetector(n_features=N_FEATURES, hidden_size=8, num_layers=1, dropout=0.0)

    return Artifacts(
        model=model, scaler=scaler, gbdt=gbdt, threshold=0.5,
        feature_names=["f_a", "f_b", "f_c"], seq_len=SEQ_LEN,
        horizon_hours=96, min_lead_hours=72,
    )


@pytest.fixture()
def client(artifacts) -> TestClient:
    return TestClient(create_app(artifacts=artifacts))


def _window(seed: int = 1) -> list[list[float]]:
    return np.random.default_rng(seed).normal(size=(SEQ_LEN, N_FEATURES)).tolist()


def test_health(client):
    r = client.get("/health")
    assert r.status_code == 200
    assert r.json() == {"status": "ok", "model_loaded": True}


def test_model_info_shapes(client):
    info = client.get("/model-info").json()
    assert info["seq_len"] == SEQ_LEN
    assert info["n_features"] == N_FEATURES
    assert info["feature_names"] == ["f_a", "f_b", "f_c"]
    assert info["horizon_hours"] == 96
    assert info["explanation_model"].startswith("hist_gbdt")


def test_predict_returns_probability_alert_and_explanation(client):
    r = client.post("/predict", json={"magnet_id": "M3", "window": _window()})
    assert r.status_code == 200, r.text
    body = r.json()

    assert body["magnet_id"] == "M3"
    assert 0.0 <= body["quench_probability"] <= 1.0
    assert body["alert"] == (body["quench_probability"] >= body["threshold"])
    assert body["threshold"] == 0.5
    assert body["min_lead_hours"] == 72

    # explanation contract: top-3 factors, sorted by |shap|, with direction
    exp = body["explanation"]
    assert len(exp) == 3
    assert all(set(f) == {"feature", "shap_value", "direction"} for f in exp)
    mags = [abs(f["shap_value"]) for f in exp]
    assert mags == sorted(mags, reverse=True)
    assert {f["feature"] for f in exp} <= {"f_a", "f_b", "f_c"}
    assert body["latency_ms"] >= 0


def test_predict_rejects_wrong_shape(client):
    bad = np.zeros((SEQ_LEN, N_FEATURES + 1)).tolist()
    r = client.post("/predict", json={"window": bad})
    assert r.status_code == 422
    assert "shape" in r.json()["detail"]

    short = np.zeros((SEQ_LEN - 1, N_FEATURES)).tolist()
    r = client.post("/predict", json={"window": short})
    assert r.status_code == 422


def test_predict_rejects_nan(client):
    # strict JSON (httpx) can't serialize NaN, so send a lax-client body
    import json as _json

    w = _window()
    w[0][0] = float("nan")
    r = client.post("/predict", content=_json.dumps({"window": w}),
                    headers={"content-type": "application/json"})
    assert r.status_code == 422
    assert "NaN" in str(r.json()["detail"])


def test_predict_rejects_missing_window(client):
    r = client.post("/predict", json={"magnet_id": "M1"})
    assert r.status_code == 422


def test_scoring_is_deterministic(client):
    w = _window(seed=7)
    a = client.post("/predict", json={"window": w}).json()
    b = client.post("/predict", json={"window": w}).json()
    assert a["quench_probability"] == b["quench_probability"]
    assert a["baseline_probability"] == b["baseline_probability"]


def test_health_degraded_when_artifacts_missing(tmp_path):
    """No model on disk -> /health reports degraded, /predict -> 503."""
    app = create_app(artifacts=None, model_dir=tmp_path)   # empty dir
    with TestClient(app, raise_server_exceptions=False) as c:
        h = c.get("/health").json()
        assert h["status"] == "degraded" and h["model_loaded"] is False
        r = c.post("/predict", json={"window": _window()})
        assert r.status_code == 503
        assert "run `python scripts/run_pipeline.py`" in r.json()["detail"]


def test_artifacts_load_from_disk(tmp_path):
    """Round-trip: save the triple to disk, Artifacts.load() serves it."""
    import joblib
    import torch

    from quenchguard.lstm import LSTMDetector, save_model

    rng = np.random.default_rng(3)
    X = rng.normal(size=(100, N_FEATURES))
    from sklearn.ensemble import HistGradientBoostingClassifier
    from sklearn.preprocessing import StandardScaler

    scaler = StandardScaler().fit(X)
    gbdt = HistGradientBoostingClassifier(max_iter=10, random_state=0)
    gbdt.fit(scaler.transform(X), (X[:, 0] > 0).astype(int))
    model = LSTMDetector(n_features=N_FEATURES, hidden_size=8, num_layers=1, dropout=0.0)

    save_model(model, tmp_path / "lstm_detector.pt",
               meta={"threshold": 0.7, "features": ["f_a", "f_b", "f_c"],
                     "seq_len": SEQ_LEN, "horizon_hours": 96})
    joblib.dump(scaler, tmp_path / "scaler.joblib")
    joblib.dump(gbdt, tmp_path / "hist_gbdt.joblib")

    art = Artifacts.load(tmp_path)
    assert art.threshold == 0.7
    assert art.seq_len == SEQ_LEN
    out = art.predict(rng.normal(size=(SEQ_LEN, N_FEATURES)))
    assert 0.0 <= out["quench_probability"] <= 1.0
    assert out["alert"] == (out["quench_probability"] >= 0.7)
    assert torch.__version__  # torch import exercised
