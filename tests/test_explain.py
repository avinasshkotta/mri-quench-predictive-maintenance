"""Tests for SHAP explanations (explain.py)."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from quenchguard.explain import (explain_case, make_explainer, shap_matrix,
                                 top_factors)

FEATURES = ["f_a", "f_b", "f_c", "f_d"]


def test_top_factors_sorted_by_absolute_impact():
    shap_row = np.array([0.4, -0.9, 0.05, -0.2])
    factors = top_factors(shap_row, FEATURES, k=3)

    assert [f["feature"] for f in factors] == ["f_b", "f_a", "f_d"]
    assert factors[0]["direction"] == "decreases risk"   # -0.9
    assert factors[1]["direction"] == "increases risk"   # +0.4
    assert all(f["shap_value"] == abs(f["shap_value"]) * (-1 if "decreases" in f["direction"] else 1)
               for f in factors)


def test_top_factors_rejects_mismatched_lengths():
    with pytest.raises(ValueError, match="feature names"):
        top_factors(np.array([0.1, 0.2]), FEATURES, k=1)


def test_top_factors_returns_at_most_k():
    factors = top_factors(np.arange(10, dtype=float), [f"f{i}" for i in range(10)], k=4)
    assert len(factors) == 4


@pytest.fixture(scope="module")
def fitted_gbdt():
    """Small HistGBDT fit on random data (the SHAP target model)."""
    from sklearn.ensemble import HistGradientBoostingClassifier

    rng = np.random.default_rng(0)
    X = rng.normal(size=(400, len(FEATURES)))
    y = (X[:, 0] + 0.5 * X[:, 1] > 0).astype(int)
    model = HistGradientBoostingClassifier(max_iter=30, random_state=0)
    model.fit(X, y)
    return model


def test_tree_shap_shapes_and_consistency(fitted_gbdt):
    X = np.random.default_rng(1).normal(size=(25, len(FEATURES)))
    explainer = make_explainer(fitted_gbdt)
    expected, values = shap_matrix(explainer, X)

    assert values.shape == X.shape
    # SHAP efficiency property on the model's raw output (log-odds for HistGBDT):
    # base value + contributions == decision function value
    from scipy.special import expit, logit

    raw_output = logit(np.clip(fitted_gbdt.predict_proba(X)[:, 1], 1e-6, 1 - 1e-6))
    assert np.allclose(expected + values.sum(axis=1), raw_output, atol=1e-4)
    # sanity: the link really is log-odds, not probability
    assert not np.allclose(expected + values.sum(axis=1),
                           expit(expected + values.sum(axis=1)), atol=1e-3)


def test_explain_case_writes_figures_and_payload(fitted_gbdt, tmp_path):
    X = np.random.default_rng(2).normal(size=(60, len(FEATURES)))
    info = explain_case(fitted_gbdt, X, FEATURES, case_index=7,
                        figures_dir=tmp_path, prefix="test")

    assert info["case_index"] == 7
    assert len(info["top_factors"]) == 3
    assert len(info["global_importance"]) == len(FEATURES)
    assert {f["feature"] for f in info["global_importance"]} == set(FEATURES)
    for fig in info["figures"]:
        assert Path(fig).exists() and Path(fig).stat().st_size > 0

    # payload must be JSON-serialisable (it lands in metrics.json)
    import json
    json.dumps(info)
