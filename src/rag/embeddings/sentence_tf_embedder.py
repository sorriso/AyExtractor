# src/rag/embeddings/sentence_tf_embedder.py — v1
"""Sentence Transformers embedding adapter (local inference).

Uses the sentence-transformers library for local embedding generation.
Models: all-MiniLM-L6-v2, multilingual-e5-large, etc.
See spec §29.3.
"""

from __future__ import annotations

import logging

from ayextractor.rag.embeddings.base_embedder import BaseEmbedder

logger = logging.getLogger(__name__)


class SentenceTransformerEmbedder(BaseEmbedder):
    """Local embeddings via sentence-transformers."""

    def __init__(
        self,
        model: str = "all-MiniLM-L6-v2",
        dimensions: int = 384,
    ) -> None:
        self._model_name = model
        self._dimensions = dimensions
        self.__model = None

    @property
    def _model(self):
        if self.__model is None:
            try:
                from sentence_transformers import SentenceTransformer
            except ImportError as e:
                raise ImportError(
                    "sentence-transformers package required: "
                    "pip install sentence-transformers"
                ) from e
            self.__model = SentenceTransformer(self._model_name)
            # Update dimensions from loaded model
            self._dimensions = self.__model.get_sentence_embedding_dimension()
        return self.__model

    async def embed_texts(self, texts: list[str]) -> list[list[float]]:
        """Embed document texts locally."""
        embeddings = self._model.encode(
            texts, show_progress_bar=False, normalize_embeddings=True
        )
        return [emb.tolist() for emb in embeddings]

    async def embed_query(self, query: str) -> list[float]:
        """Embed a single query locally."""
        embedding = self._model.encode(
            [query], show_progress_bar=False, normalize_embeddings=True
        )
        return embedding[0].tolist()

    @property
    def dimensions(self) -> int:
        return self._dimensions

    @property
    def provider_name(self) -> str:
        return "sentence_transformers"

    @property
    def model_name(self) -> str:
        return self._model_name
