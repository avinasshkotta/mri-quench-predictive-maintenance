"""Build the medallion layers: Bronze -> Silver -> Gold.

Usage
-----
    python scripts/build_medallion.py                 # all three layers
    python scripts/build_medallion.py --layer silver  # one layer (chains upstream)
    python scripts/build_medallion.py --check         # validate only, write nothing

Exit code is non-zero if any quality gate fails, so CI can run this as a
data-quality job.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from quenchguard.medallion import (  # noqa: E402
    QualityGateError,
    build_bronze,
    build_gold,
    build_silver,
    run_medallion,
)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--layer", choices=["bronze", "silver", "gold", "all"], default="all",
        help="which layer to build (default: all, in order)",
    )
    parser.add_argument(
        "--check", action="store_true",
        help="run validation gates only; do not write layer outputs",
    )
    args = parser.parse_args()

    try:
        if args.layer == "all":
            if args.check:
                build_bronze(verbose=True)
                build_silver(verbose=True, write=False)  # validate only
                print("[check] all quality gates passed (nothing written)")
            else:
                run_medallion(verbose=True)
        elif args.layer == "bronze":
            build_bronze(verbose=True)
        elif args.layer == "silver":
            build_silver(verbose=True, validate=True, write=not args.check)
        elif args.layer == "gold":
            build_gold(verbose=True)
    except QualityGateError as exc:
        print(f"\n[FAIL] {exc}", file=sys.stderr)
        return 1
    except FileNotFoundError as exc:
        print(f"\n[FAIL] {exc}", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
