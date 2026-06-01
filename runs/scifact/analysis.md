# SciFact Retrieval Analysis

## What Ran

This experiment evaluated three retrieval methods on BEIR SciFact:

- BM25 with simple lowercase regex tokenisation.
- Dense retrieval with `sentence-transformers/all-MiniLM-L6-v2`.
- Hybrid retrieval using Reciprocal Rank Fusion over BM25 and dense top-100 results.

The dataset contained 5,183 documents, 1,109 queries, and 339 binary relevance judgements in the test split. Each document was searched as:

```text
title + "\n" + text
```

The setup is intentionally simple. Documents are not chunked, BM25 uses a basic analyser, dense retrieval uses brute-force cosine similarity, and hybrid retrieval uses a fixed RRF constant of `60`.

## Results

| method | recall@5 | recall@10 | recall@100 | mrr@10 | ndcg@10 | mean ms | median ms | p95 ms | max ms |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| BM25 | 0.7268 | 0.7757 | 0.8731 | 0.6184 | 0.6523 | 9.01 | 8.11 | 15.31 | 230.88 |
| Dense | 0.7379 | 0.7833 | 0.9250 | 0.6047 | 0.6451 | 64.02 | 35.32 | 154.58 | 2341.93 |
| Hybrid | 0.7395 | 0.8059 | 0.9577 | 0.6503 | 0.6840 | 66.32 | 52.91 | 146.61 | 644.78 |

Offline costs:

| method | offline index seconds | offline embedding seconds | notes |
| --- | ---: | ---: | --- |
| BM25 | 0.25 | 0.00 | Builds a BM25 index over tokenised documents. |
| Dense | 0.00 | 43.21 | Embeds all documents on first run, then caches them. |
| Hybrid | 0.25 | 0.00 | Rebuilt the BM25 index and reused cached dense document embeddings. |

Hybrid retrieval had the best quality scores in this run. The clearest gain was at `recall@100`: hybrid reached `0.9577`, compared with `0.8731` for BM25 and `0.9250` for dense retrieval. That is the strongest evidence that BM25 and dense retrieval are finding partly different relevant documents.

The top-10 metrics show a more nuanced result. Hybrid also led on `recall@10`, `mrr@10`, and `ndcg@10`, but the margins were smaller. Dense retrieval slightly beat BM25 on recall, while BM25 slightly beat dense retrieval on `mrr@10` and `ndcg@10`. That suggests dense retrieval was better at getting relevant documents into the candidate set, while BM25 was often better at placing the first relevant result high when exact terminology mattered.

## Quality Interpretation

The result is not simply "dense beats BM25" or "hybrid beats both". The interesting pattern is that the methods make different mistakes.

BM25 preserves exact lexical evidence. Query `70`, `Activation of PPM1D suppresses p53 function.`, is a good example. BM25 ranked both labelled relevant `PPM1D` documents first and second. Dense retrieval instead ranked broader `p53` papers above the labelled relevant documents. That is a classic dense retrieval failure mode: it recognises the general topic, but softens the exact identifier that actually matters.

Dense retrieval helps when exact word overlap is not enough. Query `238`, `Cells undergoing methionine restriction may activate miRNAs.`, shows the opposite pattern. BM25 ranked methionine-heavy papers first, while dense retrieval placed the labelled microRNA paper at rank 2. The dense model picked up a more semantic relationship between the query and the relevant document.

Hybrid retrieval helps because it keeps both candidate streams alive. Query `5`, `1/2000 in UK have abnormal PrP positivity.`, is a clean case where both BM25 and dense retrieval surfaced the relevant document near the top, and hybrid ranked it first. The broader result at `recall@100` shows the same thing at scale: when the two retrievers recover different candidates, fusion improves the chance that relevant documents survive into the candidate set.

Hybrid is not a perfect reranker. Query `238` also shows that fusion can keep the relevant document in the top 5 without necessarily moving it to the best rank. Reciprocal Rank Fusion is a robust candidate-combination heuristic, not a learned relevance judge. It cannot understand the query-document pair the way a cross-encoder reranker might.

## Timing Interpretation

BM25 was much faster online in this implementation. Its mean query latency was `9.01ms`, compared with `64.02ms` for dense retrieval and `66.32ms` for hybrid retrieval.

Dense retrieval pays two costs:

- a one-time document embedding cost, which was `43.21s` on the first run;
- a per-query embedding cost, which dominated online latency.

The vector search itself was brute force. That is acceptable for 5,183 documents, but it is not how a larger retrieval system would usually be served. At larger scale, dense retrieval would need an approximate nearest neighbour index or a vector database. The timing numbers should therefore be read as local implementation measurements, not as general claims about BM25 and dense retrieval in production.

Hybrid latency was close to dense latency because it includes dense query embedding. RRF fusion itself was cheap. In other words, the extra quality from hybrid mostly costs the dense retrieval path plus a small amount of fusion work.

The max latency values also show local noise. Dense retrieval had a much higher max query latency than its median, which likely reflects model/runtime variability on the local machine. For this reason, the mean and max should not be over-interpreted without repeated benchmark runs.

## Setup Assumptions

Several choices shape the result:

- The benchmark text is already extracted and clean. The experiment does not evaluate crawling, parsing, boilerplate removal, HTML structure, tables, or chunk boundaries.
- Documents are searched as whole title-plus-abstract units. There is no passage splitting, overlap, or structure-aware chunking.
- Relevance labels are binary. A labelled document is relevant; absent query-document pairs are treated as not relevant.
- Metrics are computed from the same top-100 retrieval output. `recall@5` and `recall@10` slice the top-100 list rather than running separate retrieval jobs.
- BM25 uses simple tokenisation. A stronger analyser with stemming, phrase handling, field boosts, or domain-specific token rules could change its performance.
- Dense retrieval uses one small general-purpose sentence-transformer model. Different embedding models may shift the dense and hybrid results.
- Hybrid retrieval uses RRF with `k = 60`. This is a standard default, but not tuned for SciFact.
- No reranker is used. The experiment evaluates first-stage retrieval and rank fusion, not deeper relevance judgement.

These assumptions are acceptable for a compact local benchmark, but they limit how broadly the numbers should be interpreted.

## What The Experiment Shows

The experiment succeeds as a retrieval harness. It downloads and normalises an external benchmark, runs multiple retrieval methods through the same interface, records quality and timing, caches dense document embeddings, and produces query-level failure reports.

The substantive result is that lexical and dense retrieval provide complementary signals on SciFact. Dense retrieval improves deeper recall, BM25 remains strong when exact scientific terms matter, and hybrid retrieval benefits from combining both ranked lists. The failure cases make this clearer than the aggregate table alone.

The experiment also shows why search evaluation should not stop at one metric. Recall captures candidate coverage, MRR captures how quickly the first relevant result appears, nDCG captures top-10 ranking quality, and timing captures a separate engineering constraint. A method can improve one dimension while weakening another.

## What This Does Not Prove

This does not prove that hybrid retrieval is always best. It was best in this setup, with this dataset, this embedding model, this BM25 analyser, this fusion method, and this local runtime.

It also does not prove that dense retrieval is too slow in general. The dense implementation here uses brute-force search and local model inference. A production implementation would likely batch queries, optimise inference, and use an approximate nearest neighbour index.

Finally, the experiment does not test whether the searchable text is the right text. SciFact gives clean title and abstract fields. Real search systems often have to decide what content to extract, what to discard, and what unit should be indexed. This benchmark starts after that decision has already been made.
