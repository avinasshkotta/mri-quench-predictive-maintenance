"""SHAP explanations for the QuenchGuard models.

Tool selection (matches the XAI decision table in the course material):

* **SHAP TreeExplainer** -> ``hist_gbdt`` (tree ensemble, exact + fast).
  This is the interpretable champion for global importance, local waterfalls
  and the per-prediction "top factors" payload served by the API.
* **Gradient attribution** -> ``lstm`` (see :func:`quenchguard.lstm.gradient_importance`):
  sequences are the LSTM's native input; SHAP over 48x36 windows is orders of
  magnitude slower with no extra decision value.

Every explanation is *local to one prediction* unless stated otherwise --
global answers come from aggregating |SHAP| across the test set.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")  # headless: figures are files, never windows
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import shap  # noqa: E402


def make_explainer(model: Any) -> shap.TreeExplainer:
    """TreeExplainer for a fitted tree ensemble (HistGBDT / XGBoost / RF)."""
    return shap.TreeExplainer(model)


def shap_matrix(explainer: shap.TreeExplainer, X: np.ndarray) -> tuple[float, np.ndarray]:
    """Return (expected_value, shap_values) for a batch of rows."""
    values = explainer.shap_values(X)
    if isinstance(values, list):          # binary models may return a list
        values = values[1]
    ev = np.asarray(explainer.expected_value, dtype=float)
    expected = float(ev.reshape(-1)[0]) if ev.size else 0.0
    return expected, np.asarray(values)


def top_factors(shap_row: np.ndarray, feature_names: list[str], k: int = 3) -> list[dict]:
    """The k most influential features for ONE prediction, for API responses.

    Returned shape matches the course serving contract:
    ``[{"feature", "shap_value", "direction"}, ...]`` sorted by |SHAP|.
    """
    if len(shap_row) != len(feature_names):
        raise ValueError(f"shap_row has {len(shap_row)} values, "
                         f"but {len(feature_names)} feature names given")
    order = np.argsort(np.abs(shap_row))[::-1][:k]
    return [
        {
            "feature": str(feature_names[i]),
            "shap_value": round(float(shap_row[i]), 6),
            "direction": "increases risk" if shap_row[i] > 0 else "decreases risk",
        }
        for i in order
    ]


def save_shap_summary(shap_values: np.ndarray, X: pd.DataFrame,
                      path: Path, max_display: int = 15) -> Path:
    """Global explanation: beeswarm (direction + magnitude) for the whole test set."""
    path.parent.mkdir(parents=True, exist_ok=True)
    shap.summary_plot(shap_values, X, max_display=max_display, show=False)
    plt.tight_layout()
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close("all")
    return path


def save_shap_waterfall(expected_value: float, shap_row: np.ndarray, x_row: pd.Series,
                        feature_names: list[str], path: Path,
                        max_display: int = 10) -> Path:
    """Local explanation: why THIS window scored the way it did."""
    path.parent.mkdir(parents=True, exist_ok=True)
    explanation = shap.Explanation(
        values=shap_row,
        base_values=expected_value,
        data=x_row.values,
        feature_names=list(feature_names),
    )
    shap.plots.waterfall(explanation, max_display=max_display, show=False)
    plt.tight_layout()
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close("all")
    return path


def explain_case(model: Any, X: np.ndarray, feature_names: list[str],
                 case_index: int, figures_dir: Path,
                 prefix: str = "shap") -> dict[str, Any]:
    """One-stop helper: global summary + local waterfall + top-factors payload.

    Returns a JSON-serialisable dict that is logged into ``metrics.json`` and
    (in Phase C) echoed by the FastAPI endpoint.
    """
    explainer = make_explainer(model)
    X_df = pd.DataFrame(X, columns=feature_names)
    expected, values = shap_matrix(explainer, X)

    summary_path = save_shap_summary(values, X_df,
                                     figures_dir / f"{prefix}_beeswarm.png")
    row = values[case_index]
    waterfall_path = save_shap_waterfall(
        expected, row, X_df.iloc[case_index], feature_names,
        figures_dir / f"{prefix}_waterfall_case{case_index}.png",
    )
    return {
        "model": type(model).__name__,
        "n_explained": int(len(values)),
        "expected_value": round(expected, 6),
        "case_index": int(case_index),
        "top_factors": top_factors(row, feature_names),
        "figures": [str(summary_path), str(waterfall_path)],
        "global_importance": [
            {"feature": f, "mean_abs_shap": round(float(v), 6)}
            for f, v in sorted(
                zip(feature_names, np.abs(values).mean(axis=0)),
                key=lambda kv: kv[1], reverse=True,
            )[:15]
        ],
    }


__all__ = [
    "make_explainer", "shap_matrix", "top_factors",
    "save_shap_summary", "save_shap_waterfall", "explain_case",
]
