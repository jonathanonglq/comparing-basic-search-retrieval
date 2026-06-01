from __future__ import annotations

from dataclasses import dataclass

from src.common.timing import Timing, now
from src.retrievers.bm25 import BM25Retriever
from src.retrievers.dense import DenseRetriever


@dataclass
class HybridRetriever:
    bm25: BM25Retriever
    dense: DenseRetriever
    rrf_k: int = 60
    timing: Timing | None = None

    @classmethod
    def build(cls, dataset: str, docs: list[dict], rrf_k: int = 60) -> "HybridRetriever":
        bm25 = BM25Retriever.build(docs)
        dense = DenseRetriever.build(dataset, docs)
        timing = Timing(
            offline_index_time_seconds=bm25.timing.offline_index_time_seconds,
            offline_embedding_time_seconds=dense.timing.offline_embedding_time_seconds,
            model_load_time_seconds=dense.timing.model_load_time_seconds,
        )
        timing.extra["doc_embedding_cache_hit"] = dense.timing.extra.get("doc_embedding_cache_hit", False)
        return cls(bm25=bm25, dense=dense, rrf_k=rrf_k, timing=timing)

    def search(self, query: str, top_k: int, candidate_k: int | None = None) -> list[dict]:
        if self.timing is None:
            self.timing = Timing()
        candidate_k = candidate_k or top_k
        total_start = now()

        bm25_start = now()
        bm25_results = self.bm25.search(query, candidate_k)
        bm25_seconds = now() - bm25_start

        dense_start = now()
        dense_results = self.dense.search(query, candidate_k)
        dense_seconds = now() - dense_start

        fusion_start = now()
        scores: dict[str, float] = {}
        for results in (bm25_results, dense_results):
            for item in results:
                doc_id = str(item["doc_id"])
                scores[doc_id] = scores.get(doc_id, 0.0) + 1.0 / (self.rrf_k + int(item["rank"]))

        ranked = sorted(scores.items(), key=lambda item: item[1], reverse=True)[:top_k]
        fusion_seconds = now() - fusion_start

        self.timing.query_latencies_seconds.append(now() - total_start)
        self.timing.extra.setdefault("_bm25_query_seconds", []).append(bm25_seconds)
        self.timing.extra.setdefault("_dense_query_seconds", []).append(dense_seconds)
        self.timing.extra.setdefault("_fusion_seconds", []).append(fusion_seconds)

        return [
            {"doc_id": doc_id, "score": float(score), "rank": rank}
            for rank, (doc_id, score) in enumerate(ranked, start=1)
        ]

    def timing_dict(self) -> dict:
        from src.common.timing import latency_summary_ms

        if self.timing is None:
            return {}
        bm25_values = self.timing.extra.pop("_bm25_query_seconds", [])
        dense_values = self.timing.extra.pop("_dense_query_seconds", [])
        fusion_values = self.timing.extra.pop("_fusion_seconds", [])
        data = self.timing.to_dict()
        if bm25_values:
            data["bm25_query_mean_ms"] = latency_summary_ms(bm25_values)["query_latency_mean_ms"]
        if dense_values:
            data["dense_query_mean_ms"] = latency_summary_ms(dense_values)["query_latency_mean_ms"]
        if fusion_values:
            data["fusion_mean_ms"] = latency_summary_ms(fusion_values)["query_latency_mean_ms"]
        return data
