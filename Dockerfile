# syntax=docker/dockerfile:1.7
# Two targets:
#   api       - search API + UI (sentence-transformers, FastAPI)           docker build --target api .
#   pipeline  - api + ASR (faster-whisper) + diarization (SpeechBrain)     docker build --target pipeline .
# Models are downloaded at build time so containers start offline and reproducibly.

ARG PYTHON_VERSION=3.11
FROM python:${PYTHON_VERSION}-slim AS base
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    HF_HOME=/models/hf \
    AUDIOSEARCH_DATA_DIR=/app/data
RUN apt-get update \
 && apt-get install -y --no-install-recommends ffmpeg libpq5 curl \
 && rm -rf /var/lib/apt/lists/*
RUN pip install --no-cache-dir uv==0.5.11
WORKDIR /app
# CPU-only PyTorch keeps the image ~1.5 GB smaller than the default CUDA wheels.
RUN uv pip install --system --index-url https://download.pytorch.org/whl/cpu "torch==2.5.1" "torchaudio==2.5.1"

# ------------------------------------------------------------------------------------------------ api
FROM base AS api
COPY pyproject.toml README.md requirements.lock ./
COPY src ./src
RUN uv pip install --system -c requirements.lock ".[api]"
ARG EMBEDDING_MODEL=BAAI/bge-base-en-v1.5
ARG RERANKER_MODEL=cross-encoder/ms-marco-MiniLM-L-6-v2
RUN python -c "from sentence_transformers import SentenceTransformer, CrossEncoder; \
SentenceTransformer('${EMBEDDING_MODEL}'); CrossEncoder('${RERANKER_MODEL}')"
COPY data ./data
RUN useradd --create-home --uid 10001 app && chown -R app /app /models
USER app
ENV HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 AUDIOSEARCH_LOG_FORMAT=json
EXPOSE 8000
HEALTHCHECK --interval=15s --timeout=3s --start-period=60s --retries=3 \
  CMD curl -fsS http://localhost:8000/healthz || exit 1
CMD ["audiosearch", "serve", "--host", "0.0.0.0", "--port", "8000"]

# ------------------------------------------------------------------------------------------- pipeline
FROM api AS pipeline
USER root
RUN uv pip install --system -c requirements.lock ".[asr,diarization,eval]"
ENV HF_HUB_OFFLINE=0 TRANSFORMERS_OFFLINE=0
USER app
CMD ["audiosearch", "build"]
