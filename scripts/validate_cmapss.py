#!/usr/bin/env python
"""External validation on NASA C-MAPSS FD001 (real turbofan degradation data).

    python scripts/validate_cmapss.py --download    # fetch data + run
    python scripts/validate_cmapss.py               # run (data must exist)

Writes ``reports/external_validation.json`` and two figures under
``reports/figures/cmapss_*.png``.

This validates the *methodology* on data the authors did not generate — not
that the MRI LSTM transfers to turbofans. See
``src/quenchguard/external_validation.py`` for the protocol and its censoring
caveats.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from quenchguard.external_validation import run_external_validation   # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--download", action="store_true",
                    help="download FD001 from the NASA mirror if absent")
    ap.add_argument("--base", type=Path, default=None,
                    help="directory holding train_FD001.txt / test_FD001.txt / "
                         "RUL_FD001.txt (default: data/external/cmapss)")
    args = ap.parse_args()

    print("C-MAPSS FD001 external validation")
    report = run_external_validation(base=args.base, download=args.download)
    best = report["results"][report["champion"]]
    t = best["test"]
    print(f"\nchampion: {report['champion']} (threshold {best['threshold']:.3f})")
    print(f"  PR-AUC                : {t['window']['pr_auc']:.3f}")
    print(f"  event recall          : {t['events']['event_recall']:.0%} "
          f"({t['events']['detected']}/{t['n_observable_events']} observable)")
    print(f"  event recall in time  : {t['events']['event_recall_in_time']:.0%} "
          f"({t['events']['detected_in_time']}/{t['n_evaluable_in_time']} evaluable)")
    print(f"  false alerts          : {t['alerts']['false_alerts']} "
          f"({t['alerts']['false_per_unit']:.2f} per engine)")
    print("  -> reports/external_validation.json")
    return 0


if __name__ == "__main__":
    sys.exit(main())
