from __future__ import annotations

import math
from collections import defaultdict


def qrels_by_query(qrels: list[dict]) -> dict[str, dict[str, int]]:
    grouped: dict[str, dict[str, int]] = defaultdict(dict)
    for row in qrels:
        score = int(row["score"])
        if score > 0:
            grouped[str(row["query_id"])][str(row["doc_id"])] = score
    return dict(grouped)


def results_by_query(results: list[dict]) -> dict[str, list[dict]]:
    return {str(row["query_id"]): row["results"] for row in results}


def recall_at_k(results: list[dict], qrels: list[dict], k: int) -> float:
    relevant = qrels_by_query(qrels)
    retrieved = results_by_query(results)
    scores: list[float] = []
    for query_id, rel_docs in relevant.items():
        top_docs = {str(item["doc_id"]) for item in retrieved.get(query_id, [])[:k]}
        scores.append(len(top_docs & set(rel_docs)) / len(rel_docs))
    return sum(scores) / len(scores) if scores else 0.0


def mrr_at_k(results: list[dict], qrels: list[dict], k: int) -> float:
    relevant = qrels_by_query(qrels)
    retrieved = results_by_query(results)
    scores: list[float] = []
    for query_id, rel_docs in relevant.items():
        reciprocal = 0.0
        for rank, item in enumerate(retrieved.get(query_id, [])[:k], start=1):
            if str(item["doc_id"]) in rel_docs:
                reciprocal = 1.0 / rank
                break
        scores.append(reciprocal)
    return sum(scores) / len(scores) if scores else 0.0


def ndcg_at_k(results: list[dict], qrels: list[dict], k: int) -> float:
    relevant = qrels_by_query(qrels)
    retrieved = results_by_query(results)
    scores: list[float] = []
    for query_id, rel_docs in relevant.items():
        dcg = 0.0
        for rank, item in enumerate(retrieved.get(query_id, [])[:k], start=1):
            rel = rel_docs.get(str(item["doc_id"]), 0)
            dcg += (2**rel - 1) / math.log2(rank + 1)

        ideal_rels = sorted(rel_docs.values(), reverse=True)[:k]
        idcg = sum((2**rel - 1) / math.log2(rank + 1) for rank, rel in enumerate(ideal_rels, start=1))
        if idcg > 0:
            scores.append(dcg / idcg)
    return sum(scores) / len(scores) if scores else 0.0


def compute_metrics(results: list[dict], qrels: list[dict]) -> dict[str, float]:
    return {
        "recall@5": recall_at_k(results, qrels, 5),
        "recall@10": recall_at_k(results, qrels, 10),
        "recall@100": recall_at_k(results, qrels, 100),
        "mrr@10": mrr_at_k(results, qrels, 10),
        "ndcg@10": ndcg_at_k(results, qrels, 10),
    }
