"""FastAPI serving layer: LSTM quench scoring + SHAP factors at request time.

Contract
--------
``POST /predict`` accepts **one raw (unscaled) feature window** of shape
``seq_len x n_features`` (48 x 36 by default) — exactly what the gold layer
produces for one 15-minute step. The service:

1. scales the window with the *training* scaler persisted by the pipeline,
2. scores it with the LSTM (the champion, loaded from ``models/``),
3. computes SHAP top-3 factors from the ``hist_gbdt`` baseline on the same
   (scaled) last row, so every alert ships with a human-readable explanation.

Design notes
------------
* :func:`create_app` takes an explicit :class:`Artifacts` so tests can inject
  tiny models without touching real ``models/`` files.
* The tree explainer is built lazily on first use (TreeExplainer construction
  is cheap, but keep import-time work out of the ASGI startup path).
* Raw features go in — the caller should never need to know the scaler exists
  (training/serving skew is our problem, not the client's).
"""

from __future__ import annotations

import hashlib
import json
import os
import tempfile
import threading
import time
import warnings
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
import shap
import torch
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.preprocessing import StandardScaler

from .config import MODELS_DIR, REPORTS_DIR, default_config
from .explain import top_factors
from .features import FEATURE_COLUMNS
from .lstm import LSTMDetector

# --------------------------------------------------------------------------------------
# request / response schemas
# --------------------------------------------------------------------------------------
class PredictRequest(BaseModel):
    """One raw feature window: 48 rows x 36 features, unscaled."""

    magnet_id: str | None = Field(default=None, description="Optional asset tag, echoed back")
    window: list[list[float]] = Field(
        description=f"seq_len x n_features matrix of raw (unscaled) feature values"
    )


class Factor(BaseModel):
    feature: str
    shap_value: float
    direction: str


class PredictionResponse(BaseModel):
    magnet_id: str | None
    quench_probability: float = Field(description="P(quench within horizon_hours)")
    alert: bool = Field(description="probability >= frozen operating threshold")
    threshold: float
    horizon_hours: int
    min_lead_hours: int
    baseline_probability: float = Field(
        description="hist_gbdt probability on the same window (cross-check)"
    )
    explanation: list[Factor]
    explanation_model: str
    latency_ms: float
    model_name: str


class HealthResponse(BaseModel):
    status: str
    model_loaded: bool
    model_source: str = "local"
    model_version: str | None = None


# --------------------------------------------------------------------------------------
# operating threshold (env override) + prediction audit log (EU AI Act Art. 12)
# --------------------------------------------------------------------------------------
def _resolve_threshold(default: float) -> float:
    """Frozen artifact threshold, overridable by ``PREDICTION_THRESHOLD``.

    Week 7 deployment pattern: the decision threshold is an environment
    variable, so ops can tighten/relax it without rebuilding the image.
    """
    raw = os.environ.get("PREDICTION_THRESHOLD")
    if raw is None or not raw.strip():
        return default
    try:
        return float(raw)
    except ValueError:
        warnings.warn(f"ignoring non-numeric PREDICTION_THRESHOLD={raw!r}",
                      RuntimeWarning, stacklevel=2)
        return default


def _audit_path() -> Path | None:
    if os.environ.get("PREDICTION_LOG", "").strip().lower() in {"off", "0", "false", "none"}:
        return None
    return Path(os.environ.get("PREDICTION_LOG_PATH",
                               str(REPORTS_DIR / "prediction_log.jsonl")))


_LOG_LOCK = threading.Lock()


def append_prediction_log(record: dict[str, Any]) -> None:
    """Append one JSONL audit record per scored window (EU AI Act Art. 12).

    Fields: timestamp, magnet id, sha256 of the raw input window, probability,
    alert, threshold, top-3 SHAP factors, model name/source/version. Writing
    never breaks a prediction — failures are warnings.
    """
    path = _audit_path()
    if path is None:
        return
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        line = json.dumps(record, default=str, separators=(",", ":"))
        with _LOG_LOCK:
            with path.open("a", encoding="utf-8") as fh:
                fh.write(line + "\n")
    except Exception as exc:                       # noqa: BLE001 — never fail scoring
        warnings.warn(f"prediction audit log write failed ({exc})",
                      RuntimeWarning, stacklevel=2)


def _registry_triple(alias: str = "champion") -> tuple[Path, str]:
    """Download the champion run's ``serving/`` artifacts from the MLflow registry."""
    import mlflow
    from mlflow.tracking import MlflowClient

    from .tracking import DEFAULT_TRACKING_URI, MODEL_NAME

    mlflow.set_tracking_uri(os.environ.get("MLFLOW_TRACKING_URI", DEFAULT_TRACKING_URI))
    version = MlflowClient().get_model_version_by_alias(MODEL_NAME, alias)
    dest = tempfile.mkdtemp(prefix="quenchguard_registry_")
    mlflow.artifacts.download_artifacts(run_id=version.run_id,
                                        artifact_path="serving", dst_path=dest)
    return Path(dest) / "serving", str(version.version)


# --------------------------------------------------------------------------------------
# artifacts
# --------------------------------------------------------------------------------------
@dataclass
class Artifacts:
    """Everything a worker needs to answer /predict."""

    model: LSTMDetector
    scaler: StandardScaler
    gbdt: HistGradientBoostingClassifier
    threshold: float
    feature_names: list[str]
    seq_len: int
    horizon_hours: int
    min_lead_hours: int
    model_name: str = "quenchguard_lstm"
    source: str = "local"                  # "local" | "registry"
    model_version: str | None = None       # registry version when source=registry
    fallback_reason: str | None = None     # why registry loading fell back
    _explainer: shap.TreeExplainer | None = field(default=None, repr=False)

    @classmethod
    def load(cls, model_dir: Path | None = None,
             source: str | None = None) -> "Artifacts":
        """Load the triple the pipeline persists: LSTM + scaler + HistGBDT.

        ``source`` / ``MODEL_SOURCE`` selects the artifact origin:

        * ``local``    — ``models/`` in the repo (default; what the container ships)
        * ``registry`` — the MLflow Model Registry champion alias
          (``models:/quenchguard_lstm@champion``); falls back to ``local``
          with a warning when the registry is unreachable, so a tracked
          registry outage never takes scoring down.
        """
        source = (source or os.environ.get("MODEL_SOURCE", "local")).strip().lower()
        fallback_reason: str | None = None
        model_version: str | None = None
        if source == "registry" and model_dir is None:
            try:
                model_dir, model_version = _registry_triple()
            except Exception as exc:               # noqa: BLE001 — fall back, never fail
                fallback_reason = f"{type(exc).__name__}: {exc}"
                warnings.warn(
                    f"MODEL_SOURCE=registry unavailable ({fallback_reason}); "
                    "falling back to local models/ artifacts",
                    RuntimeWarning, stacklevel=2)
                source = "local"
        model_dir = model_dir or MODELS_DIR
        model_path = model_dir / "lstm_detector.pt"
        scaler_path = model_dir / "scaler.joblib"
        gbdt_path = model_dir / "hist_gbdt.joblib"
        missing = [p.name for p in (model_path, scaler_path, gbdt_path) if not p.exists()]
        if missing:
            raise FileNotFoundError(
                f"missing serving artifacts {missing} in {model_dir} — "
                "run `python scripts/run_pipeline.py` first"
            )

        payload = torch.load(model_path, map_location="cpu", weights_only=False)
        model = LSTMDetector(**payload["arch"])
        model.load_state_dict(payload["state_dict"])
        model.eval()

        meta = payload.get("meta", {})
        cfg = default_config()
        return cls(
            model=model,
            scaler=joblib.load(scaler_path),
            gbdt=joblib.load(gbdt_path),
            threshold=_resolve_threshold(float(meta.get("threshold", 0.5))),
            feature_names=list(meta.get("features", FEATURE_COLUMNS)),
            seq_len=int(meta.get("seq_len", cfg.window.seq_len)),
            horizon_hours=int(meta.get("horizon_hours", cfg.window.horizon_hours)),
            min_lead_hours=cfg.window.min_lead_hours,
            source=source,
            model_version=model_version,
            fallback_reason=fallback_reason,
        )

    @property
    def explainer(self) -> shap.TreeExplainer:
        if self._explainer is None:
            self._explainer = shap.TreeExplainer(self.gbdt)
        return self._explainer

    def predict(self, window: np.ndarray) -> dict[str, Any]:
        """Score one raw window; returns the JSON-serialisable response body."""
        # The production scaler was fitted on a named DataFrame; passing a bare
        # array triggers sklearn's feature-name warning (and is less safe).
        # Injected test scalers may be fitted without names, so check first.
        if getattr(self.scaler, "feature_names_in_", None) is not None:
            frame = pd.DataFrame(window, columns=self.feature_names)
            xs = self.scaler.transform(frame)                   # (seq_len, f), scaled
        else:
            xs = self.scaler.transform(window)
        with torch.no_grad():
            logit = self.model(torch.from_numpy(xs[None, ...]).float())
            prob = float(torch.sigmoid(logit).item())

        last = xs[-1:]
        baseline = float(self.gbdt.predict_proba(last)[0, 1])
        values = self.explainer.shap_values(last)
        if isinstance(values, list):
            values = values[1]
        factors = top_factors(np.asarray(values).reshape(-1), self.feature_names, k=3)

        return {
            "quench_probability": round(prob, 6),
            "alert": prob >= self.threshold,
            "threshold": round(self.threshold, 6),
            "horizon_hours": self.horizon_hours,
            "min_lead_hours": self.min_lead_hours,
            "baseline_probability": round(baseline, 6),
            "explanation": factors,
            "explanation_model": "hist_gbdt (TreeExplainer)",
            "model_name": self.model_name,
        }


# --------------------------------------------------------------------------------------
# app factory
# --------------------------------------------------------------------------------------
def create_app(artifacts: Artifacts | None = None,
               model_dir: Path | None = None) -> FastAPI:
    """Build the ASGI app. Artifacts load lazily on first /predict or /health."""
    app = FastAPI(
        title="QuenchGuard",
        description="MRI magnet quench early-warning API (LSTM + SHAP explanations)",
        version="1.0.0",
    )
    state: dict[str, Any] = {"artifacts": artifacts, "load_error": None}

    def _get() -> Artifacts:
        if state["artifacts"] is None and state["load_error"] is None:
            try:
                state["artifacts"] = Artifacts.load(model_dir)
            except Exception as exc:      # surfaced as 503, never a crash
                state["load_error"] = str(exc)
        if state["artifacts"] is None:
            raise HTTPException(status_code=503, detail=f"model unavailable: {state['load_error']}")
        return state["artifacts"]

    @app.get("/health", response_model=HealthResponse)
    def health() -> HealthResponse:
        try:
            art = _get()
            return HealthResponse(status="ok", model_loaded=True,
                                  model_source=art.source,
                                  model_version=art.model_version)
        except HTTPException:
            return HealthResponse(status="degraded", model_loaded=False)

    @app.get("/model-info")
    def model_info() -> dict[str, Any]:
        art = _get()
        return {
            "model_name": art.model_name,
            "model_source": art.source,
            "model_version": art.model_version,
            "fallback_reason": art.fallback_reason,
            "seq_len": art.seq_len,
            "n_features": len(art.feature_names),
            "feature_names": art.feature_names,
            "threshold": art.threshold,
            "horizon_hours": art.horizon_hours,
            "min_lead_hours": art.min_lead_hours,
            "explanation_model": "hist_gbdt (TreeExplainer)",
        }

    @app.post("/predict", response_model=PredictionResponse)
    def predict(req: PredictRequest) -> PredictionResponse:
        art = _get()
        window = np.asarray(req.window, dtype=np.float64)
        expected = (art.seq_len, len(art.feature_names))
        if window.ndim != 2 or window.shape != expected:
            raise HTTPException(
                status_code=422,
                detail=(f"window must have shape {expected} "
                        f"(seq_len x n_features), got {list(window.shape)}"),
            )
        # Finiteness check lives here (not in a pydantic validator) because
        # FastAPI echoes validator input back in the 422 body — which would
        # itself fail to serialize a NaN. Our detail is a plain string.
        if not bool(np.isfinite(window).all()):
            raise HTTPException(
                status_code=422,
                detail="window contains NaN or infinite values",
            )
        t0 = time.perf_counter()
        body = art.predict(window)
        body["latency_ms"] = round((time.perf_counter() - t0) * 1000, 2)
        body["magnet_id"] = req.magnet_id
        # EU AI Act Art. 12: every scored window gets an audit record
        # (input hash, output, threshold, SHAP top-3, model provenance).
        append_prediction_log({
            "ts": datetime.now(timezone.utc).isoformat(),
            "magnet_id": req.magnet_id,
            "input_sha256": hashlib.sha256(
                np.ascontiguousarray(window, dtype=np.float64).tobytes()).hexdigest(),
            "quench_probability": body["quench_probability"],
            "alert": body["alert"],
            "threshold": body["threshold"],
            "baseline_probability": body["baseline_probability"],
            "top_factors": [{"feature": f["feature"], "shap_value": f["shap_value"]}
                            for f in body["explanation"]],
            "model_name": body["model_name"],
            "model_source": art.source,
            "model_version": art.model_version,
            "latency_ms": body["latency_ms"],
        })
        return PredictionResponse(**body)

    return app


# module-level app for `uvicorn quenchguard.serving:app`
app = create_app()

__all__ = ["create_app", "app", "Artifacts", "PredictRequest", "PredictionResponse",
           "append_prediction_log"]
