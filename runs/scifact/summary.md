# scifact Retrieval Summary

Local benchmark timings are relative measurements for this implementation, not production latency claims.

| method | recall@5 | recall@10 | recall@100 | mrr@10 | ndcg@10 | mean ms | median ms | p95 ms | max ms |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| BM25 | 0.7268 | 0.7757 | 0.8731 | 0.6184 | 0.6523 | 9.01 | 8.11 | 15.31 | 230.88 |
| Dense | 0.7379 | 0.7833 | 0.9250 | 0.6047 | 0.6451 | 64.02 | 35.32 | 154.58 | 2341.93 |
| Hybrid | 0.7395 | 0.8059 | 0.9577 | 0.6503 | 0.6840 | 66.32 | 52.91 | 146.61 | 644.78 |

## Offline Costs

| method | offline index seconds | offline embedding seconds | notes |
| --- | ---: | ---: | --- |
| BM25 | 0.25 | 0.00 | Builds a BM25 index over tokenised documents. |
| Dense | 0.00 | 43.21 | Embeds all documents on first run, then caches them. |
| Hybrid | 0.25 | 0.00 | Rebuilt the BM25 index and reused cached dense document embeddings. |

## Notes

- Dense retrieval uses `sentence-transformers/all-MiniLM-L6-v2`.
- Hybrid retrieval uses Reciprocal Rank Fusion with `k = 60`.
