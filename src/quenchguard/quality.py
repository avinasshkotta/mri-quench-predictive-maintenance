"""Great Expectations quality gates for the silver layer.

Each silver table has an expectation suite encoding (a) schema/typing
contracts, (b) physics range checks, and (c) key-uniqueness constraints.
A failed expectation raises :class:`QualityGateError` and aborts the
medallion build -- bad data must never reach the gold layer.

The suites are the engineering implementation of EU AI Act Art. 10
(data governance: training data must be relevant, representative and
free of errors) and their results land in the lineage registry.
"""

from __future__ import annotations

import os
from typing import Any

# GX emits tqdm progress bars for metric calculation; silence them in logs/CI.
os.environ.setdefault("TQDM_DISABLE", "1")

import great_expectations as gx  # noqa: E402  (env var must be set first)
import pandas as pd  # noqa: E402

from .config import Config, default_config  # noqa: E402


class QualityGateError(RuntimeError):
    """Raised when a silver table fails its expectation suite."""

    def __init__(self, table: str, result: dict[str, Any]):
        self.table = table
        self.result = result
        failed = [r for r in result.get("results", []) if not r.get("success")]
        detail = "; ".join(
            f"{r['expectation']}({r.get('kwargs', {})})" for r in failed[:5]
        )
        super().__init__(
            f"Quality gate FAILED for silver table '{table}': "
            f"{result['statistics']['unsuccessful_expectations']}/"
            f"{result['statistics']['evaluated_expectations']} expectations failed. "
            f"First failures: {detail}"
        )


# --------------------------------------------------------------------------------------
# suite definitions (table name -> list of GX expectation kwargs)
# --------------------------------------------------------------------------------------
def _suites(cfg: Config) -> dict[str, list[tuple[str, dict[str, Any]]]]:
    magnet_ids = [f"MRI-{i:02d}" for i in range(1, cfg.fleet.n_magnets + 1)]
    n_rows = cfg.fleet.n_magnets * cfg.fleet.days * 24 * 60  # 1-min streams
    n_windows = cfg.fleet.n_magnets * cfg.fleet.days * 96     # 15-min windows

    return {
        "pressure": [
            ("expect_column_values_to_not_be_null",
             {"column": "magnet_id"}),
            ("expect_column_values_to_not_be_null",
             {"column": "timestamp"}),
            ("expect_column_values_to_not_be_null",
             {"column": "pressure_mpa"}),
            ("expect_column_values_to_be_in_set",
             {"column": "magnet_id", "value_set": magnet_ids}),
            ("expect_column_values_to_be_between",
             {"column": "pressure_mpa", "min_value": 0.5, "max_value": 3.5}),
            ("expect_compound_columns_to_be_unique",
             {"column_list": ["magnet_id", "timestamp"]}),
            ("expect_table_row_count_to_be_between",
             {"min_value": int(n_rows * 0.9), "max_value": int(n_rows * 1.1)}),
        ],
        "cold_head": [
            ("expect_column_values_to_not_be_null", {"column": "cold_head_k"}),
            ("expect_column_values_to_be_in_set",
             {"column": "magnet_id", "value_set": magnet_ids}),
            ("expect_column_values_to_be_between",
             {"column": "cold_head_k", "min_value": 2.0, "max_value": 20.0}),
            ("expect_compound_columns_to_be_unique",
             {"column_list": ["magnet_id", "timestamp"]}),
            ("expect_table_row_count_to_be_between",
             {"min_value": int(n_rows * 0.9), "max_value": int(n_rows * 1.1)}),
        ],
        "vibration": [
            ("expect_column_values_to_not_be_null", {"column": "rms_g"}),
            ("expect_column_values_to_be_in_set",
             {"column": "magnet_id", "value_set": magnet_ids}),
            ("expect_column_values_to_be_between",
             {"column": "rms_g", "min_value": 0.0, "max_value": 10.0}),
            ("expect_column_values_to_be_between",
             {"column": "peak_hz", "min_value": 0.0, "max_value": 500.0}),
            ("expect_column_values_to_be_between",
             {"column": "fault_ratio", "min_value": 0.0, "max_value": 1.0}),
            ("expect_compound_columns_to_be_unique",
             {"column_list": ["magnet_id", "window_end"]}),
            ("expect_table_row_count_to_be_between",
             {"min_value": int(n_windows * 0.9), "max_value": int(n_windows * 1.1)}),
        ],
        "rf": [
            ("expect_column_values_to_not_be_null", {"column": "rf_mean_w"}),
            ("expect_column_values_to_be_in_set",
             {"column": "magnet_id", "value_set": magnet_ids}),
            ("expect_column_values_to_be_between",
             {"column": "rf_mean_w", "min_value": 0.0, "max_value": 10_000.0}),
            ("expect_column_values_to_be_between",
             {"column": "readings", "min_value": 1, "max_value": 10_000}),
            ("expect_column_values_to_be_between",
             {"column": "dropout_count", "min_value": 0, "max_value": 10_000}),
            ("expect_compound_columns_to_be_unique",
             {"column_list": ["magnet_id", "window_end"]}),
            ("expect_table_row_count_to_be_between",
             {"min_value": int(n_windows * 0.9), "max_value": int(n_windows * 1.1)}),
        ],
        "events": [
            ("expect_column_values_to_not_be_null", {"column": "event_time"}),
            ("expect_column_values_to_not_be_null", {"column": "event_type"}),
            ("expect_column_values_to_be_in_set",
             {"column": "magnet_id", "value_set": magnet_ids}),
            ("expect_table_row_count_to_be_between",
             {"min_value": 1, "max_value": 10_000}),
        ],
    }


# --------------------------------------------------------------------------------------
# validation runner
# --------------------------------------------------------------------------------------
def run_silver_suite(table: str, df: pd.DataFrame,
                     cfg: Config | None = None) -> dict[str, Any]:
    """Validate ``df`` against the silver suite for ``table``.

    Returns a plain dict: ``{"success", "statistics", "results"}`` where
    ``results`` carries one record per expectation (GX result serialised to
    simple types) so it can be written straight into the lineage registry.
    """
    cfg = cfg or default_config()
    suites = _suites(cfg)
    if table not in suites:
        raise KeyError(f"No silver suite defined for '{table}'. "
                       f"Known: {sorted(suites)}")

    context = gx.get_context()
    datasource = context.data_sources.add_pandas(f"silver_{table}")
    asset = datasource.add_dataframe_asset(name=f"{table}_asset")
    batch_def = asset.add_batch_definition_whole_dataframe("whole")
    batch = batch_def.get_batch(batch_parameters={"dataframe": df})

    suite = gx.ExpectationSuite(name=f"silver_{table}_suite")
    for expectation_type, kwargs in suites[table]:
        suite.add_expectation(
            getattr(gx.expectations, _to_camel(expectation_type))(**kwargs)
        )
    context.suites.add(suite)

    validation = batch.validate(suite)
    results = [
        {
            "expectation": r.expectation_config.type,
            "kwargs": dict(r.expectation_config.kwargs),
            "success": bool(r.success),
            "observed": _summarise(r.result),
        }
        for r in validation.results
    ]
    return {
        "success": bool(validation.success),
        "statistics": dict(validation.statistics),
        "results": results,
    }


def _to_camel(snake: str) -> str:
    """``expect_column_values_to_not_be_null`` -> ``ExpectColumnValuesToNotBeNull``."""
    return "".join(part.capitalize() for part in snake.split("_"))


def _summarise(result: Any) -> Any:
    """Keep only the informative fields of a GX result object."""
    if not isinstance(result, dict):
        return str(result)
    keep = {k: result[k] for k in
            ("element_count", "unexpected_count", "unexpected_percent",
             "partial_unexpected_list", "value_set", "min_value", "max_value",
             "observed_value")
            if k in result}
    return keep


__all__ = ["run_silver_suite", "QualityGateError"]
