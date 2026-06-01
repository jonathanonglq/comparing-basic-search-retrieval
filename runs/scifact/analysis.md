# SciFact Iteration 1 Analysis

## What Ran

This run evaluated three retrieval methods on BEIR SciFact:

- BM25 with simple lowercase tokenisation.
- Dense retrieval with `sentence-transformers/all-MiniLM-L6-v2`.
- Hybrid retrieval using Reciprocal Rank Fusion over BM25 and dense top-100 results.

The dataset contained 5,183 documents, 1,109 queries, and 339 relevance judgements in the test split.

## Results

| method | recall@5 | recall@10 | recall@100 | mrr@10 | ndcg@10 | mean ms | p95 ms |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| BM25 | 0.7268 | 0.7757 | 0.8731 | 0.6184 | 0.6523 | 9.01 | 15.31 |
| Dense | 0.7379 | 0.7833 | 0.9250 | 0.6047 | 0.6451 | 64.02 | 154.58 |
| Hybrid | 0.7395 | 0.8059 | 0.9577 | 0.6503 | 0.6840 | 66.32 | 146.61 |

Hybrid retrieval performed best on the main quality metrics in this run. The biggest improvement was at Recall@100, where hybrid reached `0.9577`, compared with `0.8731` for BM25 and `0.9250` for dense retrieval. That suggests the two first-stage retrievers recover partly different relevant documents.

BM25 was much faster in this local setup. Its mean query latency was `9.01ms`, compared with `64.02ms` for dense retrieval and `66.32ms` for hybrid retrieval. These are local benchmark timings, not production latency claims. Dense and hybrid include query embedding cost, and the implementation uses brute-force vector search rather than an approximate nearest neighbour index.

## Failure Patterns

The failure report shows several useful cases:

- Dense retrieval sometimes succeeded where BM25 missed the relevant document. Query `1`, about `0-dimensional biomaterials`, is one example where the relevant document appeared in dense top 5 but not BM25 top 5.
- BM25 and dense often both found the right document, but ranked it differently. Query `3`, about rare variants and the 1,000 Genomes Project, shows BM25 ranking the relevant document first while dense put semantically adjacent but non-relevant papers above it.
- Hybrid often improved robustness by combining evidence from both lists. Query `5`, about abnormal PrP positivity in the UK, had BM25 at rank 1 and dense at rank 2; hybrid placed the relevant document first.
- Some queries failed across all three methods. Query `13`, about perinatal mortality and low birth weight, is one example where the relevant document did not appear in the top 10 for any method.

The mixed cases are the most useful. They show why this should not be reduced to "BM25 versus embeddings". BM25 preserves literal overlap well, dense retrieval can recover semantically related material, and hybrid retrieval improves candidate coverage when both signals are useful.

## What This Proves

The harness works:

- External benchmark data can be downloaded and normalised.
- BM25, dense, and hybrid retrieval can be run through the same interface.
- Metrics are computed consistently.
- Timing is recorded separately from quality.
- Dense document embeddings are cached.
- Failure inspection produces concrete examples rather than only aggregate scores.

The first result is also directionally sensible: hybrid retrieval improved recall and ranking quality, but with latency much closer to dense retrieval than BM25 because query embedding dominates the online path.

## What This Does Not Test

SciFact is already clean extracted text. It does not test whether a parser preserved the right heading, table, code block, list structure, or local context. It also does not test boilerplate, duplicate navigation text, page templates, hidden metadata, or bad chunk boundaries.

That means iteration 1 can compare retrieval methods after text has already been prepared. It cannot answer the more interesting web-search question: how much retrieval quality changes because the searchable units were parsed well or badly.

## Next Parser Question

Iteration 2 should make parser output the independent variable. The next experiment should hold the retrieval methods constant and compare how BM25, dense, and hybrid behave when the same external documents are parsed in different ways:

- naive text extraction
- boilerplate removal
- heading-aware extraction
- structure-preserving extraction for tables, lists, and code

The main question should be:

```text
Do better parser outputs improve retrieval quality enough to change which retrieval method looks strongest?
```
