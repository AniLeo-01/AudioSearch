# Common developer commands.  `make help` lists them.
PY ?= .venv/bin/python
AS ?= .venv/bin/audiosearch
SPLIT ?= test

.PHONY: help setup db index ingest build serve search test test-unit test-db eval eval-dev lint fmt docker-up docker-down clean

help:  ## show this help
	@grep -E '^[a-z-]+:.*## ' $(MAKEFILE_LIST) | awk -F':.*## ' '{printf "  %-12s %s\n", $$1, $$2}'

setup:  ## create .venv and install everything (CPU torch)
	uv venv .venv --python 3.11
	uv pip install --python $(PY) --index-url https://download.pytorch.org/whl/cpu torch==2.5.1 torchaudio==2.5.1
	uv pip install --python $(PY) -e ".[all]"

db:  ## start PostgreSQL + pgvector in Docker
	docker compose up -d db

ingest:  ## audio -> transcripts (ASR + diarization + roles); cached, idempotent
	$(AS) ingest

index:  ## transcripts -> Postgres (idempotent)
	$(AS) index

build: ingest index  ## full pipeline

serve:  ## API + UI on http://localhost:8000
	$(AS) serve --port 8000

search:  ## make search Q="why do clocks tick faster on Mars?"
	$(AS) search "$(Q)" --explain

test: test-unit test-db  ## fast test suites (no model downloads)

test-unit:  ## pure unit tests
	$(PY) -m pytest tests/unit -q

test-db:  ## SQL / API integration tests (needs Postgres)
	$(PY) -m pytest tests/integration -q

eval:  ## Recall@K regression gate on the golden test split (needs indexed corpus)
	$(PY) -m pytest tests/eval -q -s

report:  ## full evaluation report incl. ablations -> reports/
	$(AS) eval run --split $(SPLIT) --variants all --out reports

lint:  ## ruff + mypy
	$(PY) -m ruff check src tests scripts
	$(PY) -m ruff format --check src tests scripts
	$(PY) -m mypy src

fmt:  ## auto-format
	$(PY) -m ruff format src tests scripts
	$(PY) -m ruff check --fix src tests scripts

docker-up:  ## whole stack in Docker
	docker compose up --build

docker-down:
	docker compose down

clean:
	rm -rf .pytest_cache .mypy_cache .ruff_cache **/__pycache__
