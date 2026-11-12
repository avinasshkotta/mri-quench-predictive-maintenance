#!/usr/bin/env python
"""Serve the QuenchGuard API.

    python scripts/serve.py                 # http://127.0.0.1:8000
    python scripts/serve.py --port 9000 --reload

Docs at /docs (Swagger UI). Requires trained artifacts in ``models/``
(run ``python scripts/run_pipeline.py`` first).
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=8000)
    ap.add_argument("--reload", action="store_true", help="dev auto-reload")
    args = ap.parse_args()

    import uvicorn

    uvicorn.run("quenchguard.serving:app", host=args.host, port=args.port,
                reload=args.reload)
    return 0


if __name__ == "__main__":
    sys.exit(main())
