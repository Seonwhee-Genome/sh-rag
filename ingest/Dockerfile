FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PIP_DEFAULT_TIMEOUT=120 \
    PIP_RETRIES=10

WORKDIR /app
# Install dependencies in their own layer (only re-runs when pyproject.toml
# changes), then the package itself, so code/UI edits rebuild in seconds.
COPY pyproject.toml ./
RUN mkdir ingest && touch ingest/__init__.py \
 && pip install . && pip uninstall -y sh-rag-ingest && rm -rf ingest build
COPY ingest ./ingest
RUN pip install --no-deps .

ENV DOCS_DIR=/data/docs
VOLUME ["/data/docs"]
EXPOSE 8001
HEALTHCHECK --interval=15s --timeout=3s --retries=3 \
  CMD python -c "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8001/health', timeout=2)"
CMD ["python", "-m", "ingest", "serve"]
