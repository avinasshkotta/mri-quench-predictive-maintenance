#!/usr/bin/env python
"""Drift monitoring CLI: reference (train) vs current (test or supplied batch).

    python scripts/monitor.py                    # train vs test slice, write reports
    python scripts/monitor.py --check            # CI mode: exit 1 on retraining trigger
    python scripts/monitor.py --current new.parquet

Writes ``reports/drift.json`` + ``reports/drift_report.html``.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from quenchguard.features import FEATURE_COLUMNS        # noqa: E402
from quenchguard.monitoring import run_drift_check      # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--current", type=Path, default=None,
                    help="parquet with current-period features "
                         "(default: the pipeline's test split)")
    ap.add_argument("--check", action="store_true",
                    help="exit 1 when the retraining trigger fires")
    args = ap.parse_args()

    reference = current = None
    if args.current is not None:
        current = pd.read_parquet(args.current)
        cols = [c for c in FEATURE_COLUMNS if c in current.columns]
        current = current[cols]
        # reference from gold, excluding rows overlapping the current batch is
        # the caller's responsibility; here the full gold frame is the baseline
        gold = REPO_ROOT / "data" / "gold" / "features.parquet"
        if not gold.exists():
            print("gold features not found — run scripts/run_pipeline.py first", file=sys.stderr)
            return 2
        reference = pd.read_parquet(gold)

    verdict = run_drift_check(reference=reference, current=current)
    if args.check and verdict["retrain_recommended"]:
        print("DRIFT ALERT: retraining trigger fired", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
