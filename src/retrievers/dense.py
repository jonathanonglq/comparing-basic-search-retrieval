from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from sentence_transformers import SentenceTransformer

from src.common.io import ensure_dir, processed_dir, retrieval_text
from src.common.timing import Timing, latency_summary_ms, now


MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
CACHE_NAME = "all-MiniLM-L6-v2"


@dataclass
class DenseRetriever:
    docs: list[dict]
    model: SentenceTransformer
    doc_embeddings: np.ndarray
    timing: Timing

    @classmethod
    def build(cls, dataset: str, docs: list[dict], batch_size: int = 64) -> "DenseRetriever":
        timing = Timing()
        model_start = now()
        model = SentenceTransformer(MODEL_NAME)
        timing.model_load_time_seconds = now() - model_start

        cache_dir = processed_dir(dataset) / "embeddings"
        ensure_dir(cache_dir)
        embeddings_path = cache_dir / f"{CACHE_NAME}_docs.npy"
        ids_path = cache_dir / f"{CACHE_NAME}_doc_ids.json"
        expected_ids = [str(doc["doc_id"]) for doc in docs]

        cache_hit = False
        if embeddings_path.exists() and ids_path.exists():
            with ids_path.open("r", encoding="utf-8") as handle:
                cached_ids = json.load(handle)
            if cached_ids == expected_ids:
                doc_embeddings = np.load(embeddings_path)
                cache_hit = True
            else:
                doc_embeddings = cls._embed_docs(model, docs, batch_size, timing)
                cls._write_cache(embeddings_path, ids_path, doc_embeddings, expected_ids)
        else:
            doc_embeddings = cls._embed_docs(model, docs, batch_size, timing)
            cls._write_cache(embeddings_path, ids_path, doc_embeddings, expected_ids)

        timing.extra["doc_embedding_cache_hit"] = cache_hit
        return cls(docs=docs, model=model, doc_embeddings=doc_embeddings, timing=timing)

    @staticmethod
    def _write_cache(embeddings_path: Path, ids_path: Path, embeddings: np.ndarray, ids: list[str]) -> None:
        np.save(embeddings_path, embeddings)
        with ids_path.open("w", encoding="utf-8") as handle:
            json.dump(ids, handle)

    @staticmethod
    def _embed_docs(
        model: SentenceTransformer, docs: list[dict], batch_size: int, timing: Timing
    ) -> np.ndarray:
        start = now()
        texts = [retrieval_text(doc) for doc in docs]
        embeddings = model.encode(
            texts,
            batch_size=batch_size,
            show_progress_bar=True,
            normalize_embeddings=True,
            convert_to_numpy=True,
        )
        timing.offline_embedding_time_seconds = now() - start
        return embeddings

    def search(self, query: str, top_k: int) -> list[dict]:
        query_start = now()
        embed_start = now()
        query_embedding = self.model.encode(
            [query],
            show_progress_bar=False,
            normalize_embeddings=True,
            convert_to_numpy=True,
        )[0]
        embed_seconds = now() - embed_start

        search_start = now()
        scores = self.doc_embeddings @ query_embedding
        limit = min(top_k, len(scores))
        if limit == 0:
            self.timing.query_latencies_seconds.append(now() - query_start)
            return []
        candidate_idx = np.argpartition(scores, -limit)[-limit:]
        ranked_idx = candidate_idx[np.argsort(scores[candidate_idx])[::-1]]
        search_seconds = now() - search_start

        self.timing.query_latencies_seconds.append(now() - query_start)
        self.timing.extra.setdefault("_query_embedding_seconds", []).append(embed_seconds)
        self.timing.extra.setdefault("_similarity_search_seconds", []).append(search_seconds)

        return [
            {
                "doc_id": str(self.docs[int(idx)]["doc_id"]),
                "score": float(scores[int(idx)]),
                "rank": rank,
            }
            for rank, idx in enumerate(ranked_idx, start=1)
        ]

    def timing_dict(self) -> dict:
        embed_values = self.timing.extra.pop("_query_embedding_seconds", [])
        search_values = self.timing.extra.pop("_similarity_search_seconds", [])
        data = self.timing.to_dict()
        if embed_values:
            data["query_embedding_mean_ms"] = latency_summary_ms(embed_values)["query_latency_mean_ms"]
        if search_values:
            data["similarity_search_mean_ms"] = latency_summary_ms(search_values)["query_latency_mean_ms"]
        return data
