# QuenchGuard serving image: FastAPI + LSTM + SHAP explanations
#
# Build (from the repo root, after a pipeline run has produced models/):
#   docker build -t quenchguard .
# Run:
#   docker run --rm -p 8000:8000 quenchguard
#   curl http://localhost:8000/health
#
# CPU-only torch keeps the image small; the API scores one window per request
# and a GPU adds nothing at that batch size.

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
# trained serving artifacts (LSTM + scaler + HistGBDT) — run the pipeline first
COPY models/ models/

ENV PYTHONPATH=/app/src

EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=5s --start-period=20s --retries=3 \
    CMD python -c "import urllib.request,sys; sys.exit(0 if urllib.request.urlopen('http://127.0.0.1:8000/health').status==200 else 1)"

CMD ["uvicorn", "quenchguard.serving:app", "--host", "0.0.0.0", "--port", "8000"]
