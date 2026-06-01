# scifact Retrieval Summary

Local benchmark timings are relative measurements for this implementation, not production latency claims.

| method | recall@5 | recall@10 | recall@100 | mrr@10 | ndcg@10 | mean ms | p95 ms |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| bm25 | 0.7268 | 0.7757 | 0.8731 | 0.6184 | 0.6523 | 9.01 | 15.31 |
| dense | 0.7379 | 0.7833 | 0.9250 | 0.6047 | 0.6451 | 64.02 | 154.58 |
| hybrid | 0.7395 | 0.8059 | 0.9577 | 0.6503 | 0.6840 | 66.32 | 146.61 |

## Notes

- Dense retrieval uses `sentence-transformers/all-MiniLM-L6-v2`.
- Hybrid retrieval uses Reciprocal Rank Fusion with `k = 60`.
