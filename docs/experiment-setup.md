# External Retrieval Benchmark Harness

## Goal

Build a local retrieval evaluation harness on an external labelled benchmark. The purpose of this experiment is to compare lexical, dense, and hybrid retrieval under one consistent measurement loop.

The core loop is:

```text
external corpus -> normalised files -> index -> retrieve -> evaluate -> inspect failures
```

The experiment should answer:

- Can we load an external corpus, queries, and relevance labels reproducibly?
- How do Best Matching 25 (BM25), dense retrieval, and hybrid retrieval compare on the same benchmark?
- What quality-latency trade-offs show up even in a small local setup?
- Which failures look lexical, semantic, or pipeline-related?
- What does a clean benchmark fail to test about parsing?

The expected conclusion is deliberately bounded: a clean benchmark can compare retrieval algorithms, but it assumes the searchable text already exists in a useful form.

## Dataset Choice

Use BEIR SciFact for this experiment.

Reasons:

- External corpus with existing relevance labels.
- Small enough to run locally without building infrastructure first.
- Scientific claim retrieval is non-trivial, so dense retrieval and lexical retrieval can plausibly differ.
- Cleaner and easier to validate than raw web HTML.
- Compatible with the broader BEIR retrieval benchmark format.

Fallback dataset: BEIR NFCorpus. Use it only if SciFact loading is unexpectedly blocked.

Sources:

- BEIR benchmark: https://github.com/beir-cellar/beir
- SciFact BEIR zip: https://public.ukp.informatik.tu-darmstadt.de/thakur/BEIR/datasets/scifact.zip

## Non-Goals

Do not implement these in this experiment:

- Raw HTML parsing.
- Common Crawl or WARC processing.
- Web crawling.
- Parser variants.
- Reranking.
- Hierarchical Navigable Small World (HNSW) or vector database optimisation.
- GPU batching.
- Hand-labelled custom data.
- Complex analyser experiments such as stemming, stopword tuning, synonym maps, or field boosts.
- Production latency claims.

These are separate experiments. This project is about building a reliable retrieval benchmark harness.

## Repository Structure

Use the more granular structure:

```text
search_retrieval/
  README.md
  requirements.txt
  data/
    raw/
      scifact/
    processed/
      scifact/
        corpus.jsonl
        queries.jsonl
        qrels.jsonl
        embeddings/
  src/
    datasets/
      __init__.py
      load_scifact.py
    retrievers/
      __init__.py
      bm25.py
      dense.py
      hybrid.py
    eval/
      __init__.py
      metrics.py
      run_eval.py
    analysis/
      __init__.py
      inspect_failures.py
    common/
      __init__.py
      io.py
      timing.py
      text.py
  runs/
    scifact/
      bm25/
      dense/
      hybrid/
  docs/
    experiment-setup.md
```

Use Python for this experiment.

## Dependencies

Start with:

```text
numpy
rank-bm25
requests
sentence-transformers
scikit-learn
tqdm
```

Optional if useful:

```text
pandas
pytrec_eval
```

Prefer implementing the small metric functions locally at first. `pytrec_eval` is useful later, but a local implementation keeps the first pass inspectable.

## Normalised Data Formats

Keep raw downloaded or cached dataset files under:

```text
data/raw/scifact/
```

Write normalised JSON Lines files under:

```text
data/processed/scifact/
```

### Corpus

Path:

```text
data/processed/scifact/corpus.jsonl
```

Schema:

```json
{"doc_id":"123","title":"...","text":"..."}
```

Rules:

- `doc_id` must be a string.
- `title` may be empty, but the key should always exist.
- `text` should be the main document text supplied by the benchmark.
- Do not chunk documents in this experiment.
- Retrieval text is `title + "\n" + text`.

### Queries

Path:

```text
data/processed/scifact/queries.jsonl
```

Schema:

```json
{"query_id":"456","text":"..."}
```

Rules:

- `query_id` must be a string.
- `text` is the benchmark query or claim.

### Relevance Judgements

Path:

```text
data/processed/scifact/qrels.jsonl
```

Schema:

```json
{"query_id":"456","doc_id":"123","score":1}
```

Rules:

- Treat any `score > 0` as relevant for Recall and Mean Reciprocal Rank (MRR).
- Preserve the original score for Normalised Discounted Cumulative Gain (nDCG).

## Command Interface

Use module commands from the repo root.

### Prepare Data

```bash
python -m src.datasets.load_scifact
```

Responsibilities:

- Download the official BEIR SciFact zip if it is not already present.
- Write normalised `corpus.jsonl`, `queries.jsonl`, and `qrels.jsonl`.
- Print counts:
  - documents
  - queries
  - qrels
  - average document text length
  - average query length
- Print one sample query with its relevant document titles.

Success condition:

- The three processed files exist.
- Counts are non-zero.
- At least one query can be joined to its relevant document.

### Run A Retriever

```bash
python -m src.eval.run_eval --dataset scifact --retriever bm25 --top-k 100
python -m src.eval.run_eval --dataset scifact --retriever dense --top-k 100
python -m src.eval.run_eval --dataset scifact --retriever hybrid --top-k 100
```

Responsibilities:

- Load processed dataset files.
- Build or load the relevant index/cache.
- Retrieve top `k` results for every query.
- Save raw run results.
- Compute metrics.
- Save timing.
- Update the summary files.

## Retrieval Output Format

Each retriever writes:

```text
runs/scifact/<retriever>/results.jsonl
```

Schema:

```json
{
  "query_id": "456",
  "retriever": "bm25",
  "results": [
    {"doc_id": "123", "score": 12.4, "rank": 1},
    {"doc_id": "789", "score": 9.8, "rank": 2}
  ]
}
```

Rules:

- `rank` starts at `1`.
- Results must be sorted by descending score.
- Save exactly `top_k` results unless the corpus is smaller.
- Scores are retriever-specific and are not assumed to be comparable across methods.

## BM25 Retriever

File:

```text
src/retrievers/bm25.py
```

Use `rank-bm25` for this experiment.

Text preparation:

```text
retrieval_text = title + "\n" + text
```

Tokenisation:

- lowercase
- split on non-word characters
- remove empty tokens

Example helper:

```python
re.findall(r"[a-z0-9_]+", text.lower())
```

Keep this intentionally simple. More complex analyser work belongs in a separate experiment.

BM25 responsibilities:

- Build tokenised corpus.
- Build `BM25Okapi`.
- For each query, tokenise query text.
- Score all documents.
- Return top `k`.

Timing to record:

- tokenisation/index construction time
- per-query retrieval latency

## Dense Retriever

File:

```text
src/retrievers/dense.py
```

Use:

```text
sentence-transformers/all-MiniLM-L6-v2
```

Reasons:

- Fast enough locally.
- Common baseline.
- Good enough for first-pass semantic retrieval.

Embedding text:

```text
title + "\n" + text
```

Similarity:

- cosine similarity
- brute-force matrix similarity is acceptable for SciFact

Cache files:

```text
data/processed/scifact/embeddings/all-MiniLM-L6-v2_docs.npy
data/processed/scifact/embeddings/all-MiniLM-L6-v2_doc_ids.json
```

Optional query cache:

```text
data/processed/scifact/embeddings/all-MiniLM-L6-v2_queries.npy
data/processed/scifact/embeddings/all-MiniLM-L6-v2_query_ids.json
```

Dense responsibilities:

- Load cached document embeddings if present.
- Otherwise embed all documents and save cache.
- Embed queries, either cached or on demand.
- Compute cosine similarity.
- Return top `k`.

Timing to record:

- model load time
- document embedding time, if cache miss
- query embedding time
- similarity search time
- total per-query online latency

Separate offline document embedding cost from online query cost.

## Hybrid Retriever

File:

```text
src/retrievers/hybrid.py
```

Use Reciprocal Rank Fusion (RRF), not score-weighted fusion.

Reason:

- BM25 scores and cosine similarities are not calibrated to the same scale.
- RRF uses rank positions and avoids misleading score arithmetic.

Formula:

```text
rrf_score(doc) = sum(1 / (rrf_k + rank_from_system))
```

Default:

```text
rrf_k = 60
```

Hybrid responsibilities:

- Run or load BM25 top `100`.
- Run or load dense top `100`.
- Fuse per query.
- Return top `100`.

Rules:

- A document missing from one retriever contributes zero from that retriever.
- Preserve component ranks in optional debug metadata if convenient.
- The public `results.jsonl` should use the same output schema as other retrievers.

Timing to record:

- BM25 retrieval time
- dense online retrieval time
- RRF fusion time
- total hybrid latency

## Metrics

File:

```text
src/eval/metrics.py
```

Implement:

- `recall_at_k`
- `mrr_at_k`
- `ndcg_at_k`

Report:

```text
Recall@5
Recall@10
Recall@100
MRR@10
nDCG@10
```

### Recall@k

For each query:

```text
recall@k = relevant_docs_in_top_k / total_relevant_docs_for_query
```

Then average across queries with at least one relevant document.

Also consider reporting hit rate separately later, but it is not required here.

### MRR@k

For each query:

```text
MRR@k = 1 / rank_of_first_relevant_doc
```

If no relevant document appears in top `k`, score is `0`.

Average across queries with at least one relevant document.

### nDCG@k

Discounted Cumulative Gain:

```text
DCG@k = sum((2^rel_i - 1) / log2(i + 1))
```

where `i` is one-indexed rank.

Ideal DCG is computed by sorting known relevance scores for that query in descending order.

```text
nDCG@k = DCG@k / IDCG@k
```

If `IDCG@k` is zero, skip that query.

## Timing Metrics

File:

```text
src/common/timing.py
```

Collect timing as local benchmark measurements, not production latency claims.

Record:

```text
offline_index_time_seconds
offline_embedding_time_seconds
query_latency_mean_ms
query_latency_p50_ms
query_latency_p95_ms
query_latency_max_ms
query_count
```

For dense retrieval, split online time where possible:

```text
query_embedding_mean_ms
similarity_search_mean_ms
```

For hybrid retrieval, split:

```text
bm25_query_mean_ms
dense_query_mean_ms
fusion_mean_ms
```

Timing output:

```text
runs/scifact/<retriever>/timing.json
```

Summary table should include quality and timing together:

```text
method    recall@5    recall@10    recall@100    mrr@10    ndcg@10    mean_ms    p95_ms
bm25      ...
dense     ...
hybrid    ...
```

Interpretation rule:

- Use timing to compare relative local costs in this implementation.
- Do not claim general production latency from this benchmark.

## Summary Outputs

Each run writes:

```text
runs/scifact/<retriever>/metrics.json
runs/scifact/<retriever>/timing.json
```

The evaluation command should also update:

```text
runs/scifact/summary.json
runs/scifact/summary.md
```

`summary.md` should include:

- dataset name
- model name for dense retrieval
- top-k setting
- metric table
- timing table
- short notes on caveats

## Failure Inspection

File:

```text
src/analysis/inspect_failures.py
```

Command:

```bash
python -m src.analysis.inspect_failures --dataset scifact --top-k 10
```

The script should load:

```text
runs/scifact/bm25/results.jsonl
runs/scifact/dense/results.jsonl
runs/scifact/hybrid/results.jsonl
```

Generate:

```text
runs/scifact/failure_report.md
```

Inspect these categories:

- BM25 succeeds in top `10`, dense fails.
- Dense succeeds in top `10`, BM25 fails.
- Hybrid succeeds in top `10`, at least one component fails.
- All methods fail in top `10`.
- All methods succeed, but rank the first relevant document differently.

For each selected query, print:

```text
## Query: <query_id>

Query text:
...

Relevant documents:
- <doc_id> | <title>
  <short snippet>

BM25 top 5:
1. <doc_id> | score=<score> | relevant=<yes/no> | <title>
   <short snippet>

Dense top 5:
...

Hybrid top 5:
...
```

Snippet rule:

- Use the first `300` characters of document text.
- Replace newlines with spaces.
- Keep output readable rather than exhaustive.

Success condition:

- The report contains at least two examples per category where available.
- The report makes it easy to see whether failures are lexical, semantic, or shared.

## Analysis Notes

After the automated failure report is generated, write a short human-authored analysis:

```text
runs/scifact/analysis.md
```

Questions to answer:

- Which method performed best overall?
- Which method was fastest online?
- Did hybrid improve Recall@10 or Recall@100?
- Did hybrid add meaningful latency?
- Which queries did BM25 handle better?
- Which queries did dense retrieval handle better?
- Where did all methods fail?
- What does this clean benchmark fail to test?
- What parser-related question would be worth testing separately?

The last two questions should explicitly bridge to parser experiments.

Suggested closing limitation:

```text
This benchmark compares retrieval methods after text has already been extracted. It does not tell us whether the right headings, tables, code blocks, or surrounding context would survive a real web parser.
```

## Implementation Order

### Step 1: Project Skeleton

Create directories, `requirements.txt`, package `__init__.py` files, and shared I/O helpers.

Acceptance criteria:

- `python -m src.datasets.load_scifact` imports correctly, even before full implementation.

### Step 2: Dataset Loader

Implement `load_scifact.py`.

Acceptance criteria:

- Processed `corpus.jsonl`, `queries.jsonl`, and `qrels.jsonl` exist.
- Counts print successfully.
- One query can be joined to labelled relevant docs.

### Step 3: BM25 Baseline

Implement `common/text.py`, `retrievers/bm25.py`, and enough of `run_eval.py` to run BM25.

Acceptance criteria:

- `runs/scifact/bm25/results.jsonl` exists.
- Metrics and timing are saved.

### Step 4: Dense Baseline

Implement `retrievers/dense.py`.

Acceptance criteria:

- Document embeddings are cached.
- Dense results are saved.
- Re-running uses cached embeddings.

### Step 5: Hybrid RRF

Implement `retrievers/hybrid.py`.

Acceptance criteria:

- Hybrid results are saved.
- Output includes documents from both BM25 and dense candidate lists.

### Step 6: Metrics And Summary

Complete metric calculations and summary writing.

Acceptance criteria:

- `summary.json` and `summary.md` compare all completed methods.
- Quality and timing appear in one table.

### Step 7: Failure Report

Implement `inspect_failures.py`.

Acceptance criteria:

- `failure_report.md` contains concrete examples from each available category.
- It includes query text, relevant docs, and top results for BM25, dense, and hybrid.

### Step 8: Analysis

Write `runs/scifact/analysis.md`.

Acceptance criteria:

- The analysis is specific to observed results.
- It does not overclaim from local timings.
- It clearly states what this benchmark does and does not test.

## Expected Deliverables

By the end of the experiment, the repo should contain:

```text
data/processed/scifact/corpus.jsonl
data/processed/scifact/queries.jsonl
data/processed/scifact/qrels.jsonl
runs/scifact/bm25/results.jsonl
runs/scifact/dense/results.jsonl
runs/scifact/hybrid/results.jsonl
runs/scifact/bm25/metrics.json
runs/scifact/dense/metrics.json
runs/scifact/hybrid/metrics.json
runs/scifact/bm25/timing.json
runs/scifact/dense/timing.json
runs/scifact/hybrid/timing.json
runs/scifact/summary.json
runs/scifact/summary.md
runs/scifact/failure_report.md
runs/scifact/analysis.md
```

## Definition Of Done

The experiment is complete when:

- All three retrieval methods run from command line.
- Metrics are computed consistently for all methods.
- Timing is recorded separately from quality.
- Dense embeddings are cached.
- Hybrid uses Reciprocal Rank Fusion.
- The summary table compares quality and latency.
- Failure inspection identifies concrete cases where methods differ.
- The analysis states what this benchmark proves and what it cannot test.

The main limitation should be:

```text
Retrieval quality was evaluated on already-clean text. This experiment does not test whether parser choices change the searchable units enough to alter BM25, dense, and hybrid retrieval outcomes.
```
