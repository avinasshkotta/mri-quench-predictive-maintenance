#!/usr/bin/env python
"""Fleet fairness audit CLI (Fairlearn, grouped by magnet).

    python scripts/audit_fairness.py              # champion LSTM on the test split
    python scripts/audit_fairness.py --check      # exit 1 on a policy failure

Writes ``reports/fairness.json``: per-magnet selection / TPR / FPR, event-level
recall per event-bearing magnet, alert-burden disparity, per-magnet PR-AUC and
the pass/warn/fail verdict.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from quenchguard.fairness import run_fairness_audit      # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--check", action="store_true",
                    help="exit 1 when the audit verdict is 'fail'")
    args = ap.parse_args()

    audit = run_fairness_audit()
    if args.check and audit["verdict"] == "fail":
        print("FAIRNESS POLICY FAILURE", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
