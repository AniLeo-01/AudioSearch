from __future__ import annotations

import numpy as np

from . import config

QUERY_PREFIX = "Represent this sentence for searching relevant passages: "  # BGE: queries only


class Embedder:
    def __init__(self, name: str = config.EMBED_MODEL) -> None:
        from sentence_transformers import SentenceTransformer

        self.model = SentenceTransformer(name, device="cpu")

    def docs(self, texts: list[str]) -> np.ndarray:
        return self.model.encode(texts, batch_size=32, normalize_embeddings=True, convert_to_numpy=True)

    def query(self, text: str) -> np.ndarray:
        return self.docs([QUERY_PREFIX + text])[0]

    def lexicon(self) -> frozenset[str]:
        """Whole words in the WordPiece vocab (~20k): a free 'common English word' list."""
        vocab = self.model.tokenizer.get_vocab()
        return frozenset(t for t in vocab if t.isalpha() and t.islower() and len(t) >= 3)
