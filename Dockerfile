# QuenchGuard image: customer web UI (Next.js) + streaming agent API + LSTM serving
#
# Build (from the repo root; models/ and frontend/package-lock.json committed):
#   docker build -t quenchguard .
# Run:
#   docker run --rm -p 8000:8000 -e OPENAI_API_KEY=sk-... quenchguard
#   open http://localhost:8000          # customer UI
#   curl http://localhost:8000/health   # model health
#
# CPU-only torch keeps the runtime small; the API scores one window per request
# and a GPU adds nothing at that batch size.

# ---- stage 1: build the Next.js static export -------------------------------------------
FROM node:22-slim AS ui

WORKDIR /ui
COPY frontend/package.json frontend/package-lock.json ./
RUN npm ci
COPY frontend/ ./
RUN npm run build                      # static export -> /ui/out

# ---- stage 2: python runtime ------------------------------------------------------------
FROM python:3.12-slim

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1

WORKDIR /app

# torch first (CPU wheels) so this layer is cached across code changes
RUN pip install torch --index-url https://download.pytorch.org/whl/cpu

COPY requirements.txt .
RUN pip install -r requirements.txt

COPY src/ src/
COPY scripts/ scripts/
COPY docs/ docs/
COPY README.md .
# trained serving artifacts (LSTM + scaler + HistGBDT) — committed in the repo
COPY models/ models/
# committed demo data bundle so /api/fleet and telemetry work without a pipeline run
COPY data/demo/ data/demo/
# customer UI static export from stage 1 (served by quenchguard.webui at /)
COPY --from=ui /ui/out frontend/out

ENV PYTHONPATH=/app/src

EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=5s --start-period=20s --retries=3 \
    CMD python -c "import urllib.request,sys; sys.exit(0 if urllib.request.urlopen('http://127.0.0.1:8000/health').status==200 else 1)"

CMD ["uvicorn", "quenchguard.webui:app", "--host", "0.0.0.0", "--port", "8000"]
