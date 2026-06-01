# Search Retrieval Experiment

This project compares three ways of retrieving relevant documents from an external labelled search benchmark:

- **BM25**, a lexical keyword retriever.
- **Dense retrieval**, an embedding-based semantic retriever.
- **Hybrid retrieval**, which combines BM25 and dense retrieval with Reciprocal Rank Fusion.

The goal is to make the retrieval pipeline measurable and inspectable. Instead of only asking which method has the highest score, the experiment records quality metrics, local latency, and concrete failure cases where the methods behave differently.

Search systems usually need both exact words and semantic similarity. BM25 is strong when the query contains terms that should match literally: identifiers, acronyms, technical phrases, numbers, error codes, and domain-specific vocabulary. Dense retrieval is useful when the user and the document describe the same idea with different words. Hybrid retrieval keeps both signals alive by merging the ranked lists from both methods.

This experiment tests that trade-off on **BEIR SciFact**, an external retrieval benchmark with real documents, queries, and relevance labels. SciFact provides clean extracted text, so the experiment focuses on retrieval rather than crawling or parsing, which would be explored in a later experiment.

The pipeline is:

```text
SciFact corpus -> normalised JSONL files -> retrieval runs -> metrics -> failure inspection
```

## What The Experiment Does

### 1. Downloads And Normalises SciFact

The dataset loader downloads the official BEIR SciFact zip and converts it into simple local JSONL files with the loader implemented in `src/datasets/load_scifact.py`.

- `corpus.jsonl`: the searchable documents.
- `queries.jsonl`: the search queries.
- `qrels.jsonl`: relevance labels saying which documents are correct for each query.

The completed run used:

```text
documents: 5183
queries: 1109
qrels: 339
average_document_tokens: 225.2
average_query_tokens: 12.9
```

### Data Format

The raw BEIR SciFact corpus is already distributed as JSONL inside the downloaded zip. A raw corpus row looks like:

```json
{
  "_id": "31715818",
  "title": "New opportunities: the use of nanotechnologies to manipulate and track stem cells.",
  "text": "Nanotechnologies are emerging platforms that could be useful...",
  "metadata": {}
}
```

The loader normalises it into the simpler schema used by this project:

```json
{
  "doc_id": "31715818",
  "title": "New opportunities: the use of nanotechnologies to manipulate and track stem cells.",
  "text": "Nanotechnologies are emerging platforms that could be useful..."
}
```

Queries follow the same pattern. Raw BEIR rows use `_id`; the processed file uses `query_id`:

```json
{
  "query_id": "1",
  "text": "0-dimensional biomaterials show inductive properties."
}
```

The relevance labels start as a BEIR TSV file:

```text
query-id    corpus-id    score
1           31715818     1
```

They are normalised into JSONL:

```json
{
  "query_id": "1",
  "doc_id": "31715818",
  "score": 1
}
```

For this SciFact run, relevance is binary: `1` means relevant and `0` means not relevant. The qrels file stores the labelled relevant pairs with `score: 1`; query-document pairs that are absent from qrels are treated as `0` during evaluation.

### 2. Runs BM25 Retrieval

BM25 is the keyword baseline. It takes each document as:

```text
title + "\n" + text
```

Then it lowercases and tokenises the text with a simple regex before building a BM25 index using `src/retrievers/bm25.py`.

Outputs:

- runs/scifact/bm25/results.jsonl
- runs/scifact/bm25/metrics.json
- runs/scifact/bm25/timing.json

`results.jsonl` contains one row per query. Each row stores the query ID, retriever name, and the ranked top documents returned by BM25:

```json
{
  "query_id": "1",
  "retriever": "bm25",
  "results": [
    {"doc_id": "10608397", "score": 10.1725, "rank": 1},
    {"doc_id": "43385013", "score": 9.8616, "rank": 2},
    ...
  ]
}
```

`metrics.json` contains aggregate retrieval quality scores for the whole query set, such as `recall@10`, `mrr@10`, and `ndcg@10`.

`timing.json` contains local timing measurements, including index construction time, mean query latency, p50 latency, p95 latency, max latency, and query count.

### 3. Runs Dense Retrieval

Dense retrieval embeds documents and queries using: ```sentence-transformers/all-MiniLM-L6-v2```

Each query, converted into an embedding, is compared against document embeddings with cosine similarity. Since SciFact is small, this implementation uses brute-force vector search rather than an approximate nearest neighbour index. This is implemented in ```src/retrievers/dense.py```.

Document embeddings are cached in the following files:

```text
data/processed/scifact/embeddings/all-MiniLM-L6-v2_docs.npy
data/processed/scifact/embeddings/all-MiniLM-L6-v2_doc_ids.json
```

The `.npy` file is a NumPy binary file containing the document embedding matrix of the form: ```(5183 documents, 384 embedding dimensions)```. The `doc_ids.json` file stores the document IDs and tells us which document that vector belongs to.

Outputs are produced in a similar format to the BM25 run in the folder ```runs/scifact/dense/*```.

### 4. Runs Hybrid Retrieval

Hybrid retrieval runs BM25 and dense retrieval, then merges their ranked lists with Reciprocal Rank Fusion. This avoids directly adding BM25 scores and cosine similarities, which are not naturally comparable.

The scoring rule used is:

```text
rrf_score = sum(1 / (60 + rank))
```

Again, the Implementation is found in ```src/retrievers/hybrid.py``` and outputs are produced in a similar format in the folder ```runs/scifact/hybrid/*```.

### 5. Computes Metrics And Timings

The evaluation runner executes a retriever over every query, saves the top results, computes quality metrics, and records local timing. The relevant scripts are:
- ```src/eval/run_eval.py```
- ```src/eval/metrics.py```
- ```src/common/timing.py```

Metrics:
- `recall@5`, `recall@10`, `recall@100`: the fraction of labelled relevant documents retrieved within the top `k` results. Higher means the retriever is less likely to miss relevant documents.
- `mrr@10`: Mean Reciprocal Rank within the top 10. This rewards putting the first relevant document near the top; rank 1 gets `1.0`, rank 2 gets `0.5`, and missing results get `0`. It only measures the first relevant hit, so queries with many relevant documents can be easier to score well on. This should be interpreted alongside recall and nDCG.
- `ndcg@10`: Normalised Discounted Cumulative Gain within the top 10. This compares the retriever's ranking against the best possible order implied by the relevance labels. Lower-ranked relevant documents are discounted. In this SciFact setup, labels are binary, so the ideal order places all labelled relevant documents before non-relevant documents. Any order among the relevant documents is equally ideal.

Timing:

- mean query latency: average online retrieval time per query.
- p50 query latency: median query latency.
- p95 query latency: latency below which 95% of queries completed.
- max query latency: slowest observed query.
- offline index time: one-time time spent building the BM25 index.
- offline embedding time: one-time time spent embedding all documents for dense retrieval.

Summary outputs:
- ```runs/scifact/summary.json```
- ```runs/scifact/summary.md```

## 6. Results

The completed SciFact run produced:

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

Hybrid retrieval performed best on the main quality metrics, especially `recall@100`. That suggests BM25 and dense retrieval recover partly different relevant documents. Combining their ranked lists gives the system a better candidate pool.

BM25 was much faster in this local implementation. Dense and hybrid retrieval both include query embedding cost, and dense similarity search is brute force. We note that these timings are useful for comparing this local setup, but are not production latency claims.

### 7. Inspects Failure Cases

Aggregate metrics are useful, but they hide the actual reasons methods differ. The failure inspection script compares BM25, dense, and hybrid results query by query. This was implemented in ```src/analysis/inspect_failures.py```, with the detailed report in ```runs/scifact/failure_report.md```.

The report samples cases where:

- dense retrieval succeeds but BM25 fails;
- BM25 succeeds but dense retrieval fails;
- hybrid succeeds when one component fails;
- all methods fail;
- all methods succeed but rank the relevant document differently.

Samples of failure cases:

- Query `238`: `Cells undergoing methionine restriction may activate miRNAs.`
  - Relevant document: `2251426`, `microRNAs: A Safeguard against Turmoil?`
  - BM25 ranks methionine-heavy papers first and misses the relevant document in the top 5.
  - Dense retrieval puts the relevant microRNA paper at rank 2, showing how semantic similarity can recover a document when exact lexical overlap is not enough.
  - Hybrid places the relevant document at rank 4 because it combines dense's microRNA signal with BM25's lexical signal, but it also shows that fusion is not always a better reranker.

- Query `5`: `1/2000 in UK have abnormal PrP positivity.`
  - Relevant document: `13734012`, about abnormal prion protein in human appendixes.
  - BM25 ranks the relevant document first because exact terms like `PrP`, `abnormal`, and `UK` are strong lexical evidence.
  - Dense retrieval ranks it second, behind a broader UK health survey result.
  - Hybrid ranks it first, because both retrievers surfaced the relevant document near the top.

- Query `70`: `Activation of PPM1D suppresses p53 function.`
  - Relevant documents: `5956380`, about gain-of-function `PPM1D` mutations in brainstem gliomas, and `4414547`, about mosaic `PPM1D` mutations.
  - BM25 ranks both relevant documents first and second because the exact tokens `PPM1D` and `p53` are strong lexical signals.
  - Dense retrieval ranks broader `p53` papers above the labelled relevant documents, so it misses the relevant documents in the top 5.
  - Hybrid recovers both relevant documents in the top 5 because BM25 contributes the exact identifier signal.


## Learning Points

- Hybrid retrieval improved both recall and ranking quality in this run. That is the main empirical result.

- The more useful lesson is that BM25 and dense retrieval fail differently. BM25 can rank exact lexical matches strongly, but it can miss semantically related documents when the words do not line up. Dense retrieval can recover semantic matches, but it can also rank broadly related non-relevant papers above the labelled relevant document. Hybrid retrieval helps because it does not force the system to choose one signal too early.

- Latency also matters. BM25 was much cheaper locally. Dense retrieval paid the cost of query embedding, and this implementation used brute-force similarity search. In a larger system, that trade-off would motivate batching, caching, approximate nearest neighbour search, or a multi-stage retrieval pipeline.

## Setup

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Run The Experiment

Prepare SciFact:

```bash
python -m src.datasets.load_scifact
```

Run retrievers:

```bash
python -m src.eval.run_eval --dataset scifact --retriever bm25 --top-k 100
python -m src.eval.run_eval --dataset scifact --retriever dense --top-k 100
python -m src.eval.run_eval --dataset scifact --retriever hybrid --top-k 100
```

Inspect failures:

```bash
python -m src.analysis.inspect_failures --dataset scifact --top-k 10
```

Generated reports:

- [Summary](runs/scifact/summary.md)
- [Failure report](runs/scifact/failure_report.md)
- [Analysis notes](runs/scifact/analysis.md)
