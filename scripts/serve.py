#!/usr/bin/env python
"""Serve QuenchGuard: the customer web API + UI and the model API.

    python scripts/serve.py                 # http://127.0.0.1:8000
    python scripts/serve.py --port 9000 --reload

The app (``quenchguard.webui:app``) serves:

* ``/``            — the Next.js customer UI (when ``frontend/out`` exists)
* ``/api/*``       — streaming agent chat (SSE), fleet/magnet panels
* ``/predict``     — raw model scoring with SHAP factors (JSON API)

Requires trained artifacts in ``models/`` (``python scripts/run_pipeline.py``)
and ``OPENAI_API_KEY`` in the environment or repo ``.env`` for the chat.
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

    uvicorn.run("quenchguard.webui:app", host=args.host, port=args.port,
                reload=args.reload)
    return 0


if __name__ == "__main__":
    sys.exit(main())
