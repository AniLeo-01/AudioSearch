"""Runtime configuration (12-factor: every knob is an ``AUDIOSEARCH_*`` environment variable)."""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import Literal

from pydantic import Field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

REPO_ROOT = Path(__file__).resolve().parents[2]


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_prefix="AUDIOSEARCH_", env_file=".env", extra="ignore")

    # --- storage -------------------------------------------------------------------------------
    database_url: str = "postgresql://audiosearch:audiosearch@localhost:5432/audiosearch"
    db_schema: str = Field(default="public", pattern=r"^[a-z_][a-z0-9_]{0,62}$")
    db_pool_min: int = 1
    db_pool_max: int = 8
    db_statement_timeout_ms: int = 5000

    # --- data layout ---------------------------------------------------------------------------
    data_dir: Path = REPO_ROOT / "data"
    transcript_set: str = "large-v3-turbo"  # sub-folder of data/transcripts holding pipeline output

    # --- ASR (faster-whisper) --------------------------------------------------------------------
    asr_model: str = "large-v3-turbo"
    asr_device: str = "cpu"
    asr_compute_type: str = "int8"
    asr_beam_size: int = 5
    asr_cpu_threads: int = 0  # 0 = library default

    # --- diarization -------------------------------------------------------------------------------
    diarization_model: str = "speechbrain/spkrec-ecapa-voxceleb"
    num_speakers: int = 2
    diar_window_sec: float = 1.5
    diar_hop_sec: float = 0.75
    diar_switch_penalty: float = 4.0  # Viterbi cost of a speaker change mid-sentence
    diar_boundary_discount: float = 0.15  # multiplier on that cost at sentence ends / long pauses

    # --- chunking ----------------------------------------------------------------------------------
    chunk_target_words: int = 70
    chunk_stride_words: int = 35
    chunk_context: Literal["none", "question", "question+title"] = "question"
    chunk_context_max_words: int = 40

    # --- embeddings / reranker -----------------------------------------------------------------------
    embedding_model: str = "BAAI/bge-small-en-v1.5"
    embedding_device: str = "cpu"
    embedding_batch_size: int = 32
    reranker_model: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"
    rerank: bool = False
    rerank_top_n: int = 30

    # --- retrieval ---------------------------------------------------------------------------------
    candidates_per_channel: int = 50
    rrf_k: int = 60
    adaptive_fusion: bool = True
    phonetic_expansion: bool = True
    hnsw_ef_search: int = 100
    nms_gap_sec: float = 10.0
    default_k: int = 10
    max_k: int = 50

    # --- API ---------------------------------------------------------------------------------------
    api_cors_origins: list[str] = ["*"]
    max_query_chars: int = 512

    # --- logging -----------------------------------------------------------------------------------
    log_level: str = "INFO"
    log_format: Literal["console", "json"] = "console"

    @field_validator("data_dir", mode="before")
    @classmethod
    def _resolve_data_dir(cls, v: str | Path) -> Path:
        p = Path(v)
        return p if p.is_absolute() else (REPO_ROOT / p)

    @property
    def audio_dir(self) -> Path:
        return self.data_dir / "audio"

    @property
    def manifest_path(self) -> Path:
        return self.data_dir / "manifest.yaml"

    @property
    def transcripts_dir(self) -> Path:
        return self.data_dir / "transcripts" / self.transcript_set

    @property
    def reference_dir(self) -> Path:
        return self.data_dir / "reference"

    @property
    def golden_queries_path(self) -> Path:
        return self.data_dir / "eval" / "queries.yaml"


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    return Settings()
