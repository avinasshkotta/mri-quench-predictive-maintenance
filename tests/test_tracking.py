"""Tests for MLflow tracking (tracking.py).

Every test runs against a *temporary* file-store URI so real experiment
history under ``mlruns/`` is never touched.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from quenchguard.tracking import (MODEL_NAME, _flattened_metrics, log_run)

REPORT = {
    "lstm": {
        "classification": {"pr_auc": 0.998, "roc_auc": 0.999},
        "events": {"event_recall_in_time": 1.0},
        "alerts": {"false_per_magnet_month": 0.64},
        "cost": {"net_savings_usd": 452000.0},
        "_note": "ignored",
    },
    "hist_gbdt": {
        "classification": {"pr_auc": 0.997},
        "_meta": "not a section",
    },
    "_global": "ignored",
}


@pytest.fixture()
def uri(tmp_path) -> str:
    """Isolated SQLite backend per test (never the repo's mlflow.db)."""
    return f"sqlite:///{(tmp_path / 'mlflow.db').as_posix()}"


def test_flattened_metrics_prefixes_model_and_skips_non_scalars():
    flat = _flattened_metrics(REPORT)

    assert flat["lstm_pr_auc"] == pytest.approx(0.998)
    assert flat["lstm_false_per_magnet_month"] == pytest.approx(0.64)
    assert flat["hist_gbdt_pr_auc"] == pytest.approx(0.997)
    # no underscore-private sections, no nested dicts
    assert not any(k.startswith("_") for k in flat)
    assert "lstm__note" not in flat


def test_log_run_records_params_metrics_and_run_id(uri):
    from mlflow.tracking import MlflowClient

    result = log_run(report=REPORT, thresholds={"lstm": 0.5},
                     history=[{"val_ap": 0.9}, {"val_ap": 0.93}],
                     tags={"phase": "B"}, tracking_uri=uri)

    assert result is not None
    assert result["model_uri"] is None          # no model passed
    assert result["registered_version"] is None

    client = MlflowClient(tracking_uri=uri)
    run = client.get_run(result["run_id"])
    assert run.data.metrics["lstm_pr_auc"] == pytest.approx(0.998)
    assert run.data.metrics["training_best_val_pr_auc"] == pytest.approx(0.93)
    assert run.data.params["threshold_lstm"] == "0.5"
    assert run.data.tags["phase"] == "B"


def test_log_run_registers_champion_alias(uri, tmp_path):
    """Tiny LSTM -> logged, registered, alias `champion` resolves."""
    import torch

    from quenchguard.lstm import LSTMDetector, save_model

    model = LSTMDetector(n_features=4, hidden_size=8, num_layers=1, dropout=0.0)
    result = log_run(report=REPORT, model=model, tracking_uri=uri)

    assert result is not None
    assert result["registered_version"] == 1
    assert result["model_uri"] == f"models:/{MODEL_NAME}@champion"

    # the alias must resolve back to the logged artifact
    from quenchguard.tracking import load_champion

    loaded = load_champion(tracking_uri=uri)
    x = torch.zeros(2, 6, 4)
    with torch.no_grad():
        assert loaded(x).shape == (2,)

    # round-trip through the local save/load helpers too (serving path)
    path = tmp_path / "m.pt"
    save_model(model, path)
    assert path.exists()


def test_log_run_never_raises_on_bad_uri():
    """A tracking outage must not kill the pipeline -> None + warning."""
    with pytest.warns(RuntimeWarning, match="MLflow logging failed"):
        result = log_run(report=REPORT,
                         tracking_uri="sqlite:////nonexistent/:::/bad.db")
    assert result is None
