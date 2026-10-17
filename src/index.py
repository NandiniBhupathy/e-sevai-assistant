"""Build and load FAISS and BM25 indexes."""
from __future__ import annotations

import json
import logging
import pickle
import re
from pathlib import Path

import faiss
import numpy as np
from rank_bm25 import BM25Okapi
from sentence_transformers import SentenceTransformer

logger = logging.getLogger(__name__)


def tokenize(text: str) -> list[str]:
    """Lowercase while preserving hyphenated service codes and Tamil characters."""
    return re.findall(r"[a-z0-9]+(?:-[a-z0-9]+)*|[\u0B80-\u0BFF]+", text.lower())


class SearchIndex:
    """Combined embedding model, FAISS index, BM25 corpus and chunk metadata."""

    def __init__(self, config):
        self.config = config
        self.model = self._load_model(config.raw["embedding"]["model"])
        self.chunks = []
        self.faiss = None
        self.bm25 = None
        self.tokens = []

    @staticmethod
    def _load_model(model_name: str) -> SentenceTransformer:
        """Load the embedding model with a single-copy weight load.

        The default Hugging Face path materialises the state dict and then the
        model, so a 2.3 GB checkpoint briefly needs roughly twice its size and
        gets the process killed on 8 GB machines. `low_cpu_mem_usage` maps the
        checkpoint and loads it straight into the model instead.
        """
        return SentenceTransformer(model_name, model_kwargs={"low_cpu_mem_usage": True})

    def build(self, chunks: list[dict]) -> None:
        self.chunks = chunks
        texts = [c["text"] for c in chunks]
        vectors = self.model.encode(
            texts,
            batch_size=int(self.config.raw["embedding"]["batch_size"]),
            normalize_embeddings=bool(self.config.raw["embedding"]["normalize"]),
            show_progress_bar=True,
        ).astype("float32")
        self.faiss = faiss.IndexFlatIP(vectors.shape[1])
        self.faiss.add(vectors)
        self.tokens = [tokenize(t) for t in texts]
        self.bm25 = BM25Okapi(self.tokens)
        self.save()

    def save(self) -> None:
        self.config.index_dir.mkdir(parents=True, exist_ok=True)
        faiss.write_index(self.faiss, str(self.config.faiss_file))
        self.config.chunks_file.write_text(
            "\n".join(json.dumps(c, ensure_ascii=False) for c in self.chunks) + "\n",
            encoding="utf-8",
        )
        with self.config.bm25_file.open("wb") as handle:
            pickle.dump(self.bm25, handle)

    def load(self) -> None:
        if not self.config.faiss_file.exists() or not self.config.chunks_file.exists() or not self.config.bm25_file.exists():
            raise FileNotFoundError("Index missing. Run scripts/build_index.py.")
        self.faiss = faiss.read_index(str(self.config.faiss_file))
        self.chunks = [
            json.loads(line) for line in self.config.chunks_file.read_text(encoding="utf-8").splitlines() if line.strip()
        ]
        self.tokens = [tokenize(c["text"]) for c in self.chunks]
        with self.config.bm25_file.open("rb") as handle:
            self.bm25 = pickle.load(handle)

    def dense_search(self, query: str, k: int = 20):
        vector = self.model.encode(
            [query],
            normalize_embeddings=bool(self.config.raw["embedding"]["normalize"]),
        ).astype("float32")
        scores, ids = self.faiss.search(vector, min(k, len(self.chunks)))
        return [(int(i), float(score)) for i, score in zip(ids[0], scores[0]) if i >= 0]

    def bm25_search(self, query: str, k: int = 20):
        scores = self.bm25.get_scores(tokenize(query))
        ids = np.argsort(scores)[::-1][:min(k, len(scores))]
        return [(int(i), float(scores[i])) for i in ids if scores[i] > 0]
