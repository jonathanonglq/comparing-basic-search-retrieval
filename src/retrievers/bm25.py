from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from rank_bm25 import BM25Okapi

from src.common.io import retrieval_text
from src.common.text import tokenize
from src.common.timing import Timing, now


@dataclass
class BM25Retriever:
    docs: list[dict]
    timing: Timing
    _bm25: BM25Okapi

    @classmethod
    def build(cls, docs: list[dict]) -> "BM25Retriever":
        timing = Timing()
        start = now()
        tokenized = [tokenize(retrieval_text(doc)) for doc in docs]
        bm25 = BM25Okapi(tokenized)
        timing.offline_index_time_seconds = now() - start
        return cls(docs=docs, timing=timing, _bm25=bm25)

    def search(self, query: str, top_k: int) -> list[dict]:
        start = now()
        scores = self._bm25.get_scores(tokenize(query))
        limit = min(top_k, len(scores))
        if limit == 0:
            self.timing.query_latencies_seconds.append(now() - start)
            return []
        candidate_idx = np.argpartition(scores, -limit)[-limit:]
        ranked_idx = candidate_idx[np.argsort(scores[candidate_idx])[::-1]]
        results = [
            {
                "doc_id": str(self.docs[int(idx)]["doc_id"]),
                "score": float(scores[int(idx)]),
                "rank": rank,
            }
            for rank, idx in enumerate(ranked_idx, start=1)
        ]
        self.timing.query_latencies_seconds.append(now() - start)
        return results
