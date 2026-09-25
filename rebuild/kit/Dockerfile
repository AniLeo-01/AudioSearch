# API image: serves search over the committed transcripts (ASR/diarization run offline with `audiosearch ingest`).
FROM python:3.11-slim
RUN pip install --no-cache-dir torch==2.5.1 --index-url https://download.pytorch.org/whl/cpu
WORKDIR /app
COPY pyproject.toml ./
COPY src ./src
RUN pip install --no-cache-dir ".[api]" \
 && python -c "from sentence_transformers import SentenceTransformer; SentenceTransformer('BAAI/bge-base-en-v1.5')"
COPY data ./data
ENV AUDIOSEARCH_DATA=/app/data
EXPOSE 8000
CMD ["sh", "-c", "audiosearch init-db && audiosearch index && audiosearch serve"]
