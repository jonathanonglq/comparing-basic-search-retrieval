from __future__ import annotations

import argparse
from collections import defaultdict

from src.common.io import load_processed_dataset, read_jsonl, retrieval_text, runs_dir
from src.common.text import snippet
from src.eval.metrics import qrels_by_query


METHODS = ["bm25", "dense", "hybrid"]


def load_runs(dataset: str) -> dict[str, dict[str, list[dict]]]:
    runs: dict[str, dict[str, list[dict]]] = {}
    for method in METHODS:
        path = runs_dir(dataset) / method / "results.jsonl"
        rows = read_jsonl(path)
        runs[method] = {str(row["query_id"]): row["results"] for row in rows}
    return runs


def succeeds(results: list[dict], relevant: set[str], top_k: int) -> bool:
    return any(str(item["doc_id"]) in relevant for item in results[:top_k])


def first_relevant_rank(results: list[dict], relevant: set[str], top_k: int) -> int | None:
    for rank, item in enumerate(results[:top_k], start=1):
        if str(item["doc_id"]) in relevant:
            return rank
    return None


def categorise(runs: dict[str, dict[str, list[dict]]], qrels: dict[str, dict[str, int]], top_k: int) -> dict[str, list[str]]:
    categories: dict[str, list[str]] = defaultdict(list)
    for query_id, rel_docs in qrels.items():
        relevant = set(rel_docs)
        bm25_ok = succeeds(runs["bm25"].get(query_id, []), relevant, top_k)
        dense_ok = succeeds(runs["dense"].get(query_id, []), relevant, top_k)
        hybrid_ok = succeeds(runs["hybrid"].get(query_id, []), relevant, top_k)

        if bm25_ok and not dense_ok:
            categories["BM25 succeeds, dense fails"].append(query_id)
        if dense_ok and not bm25_ok:
            categories["Dense succeeds, BM25 fails"].append(query_id)
        if hybrid_ok and (not bm25_ok or not dense_ok):
            categories["Hybrid succeeds when a component fails"].append(query_id)
        if not bm25_ok and not dense_ok and not hybrid_ok:
            categories["All methods fail"].append(query_id)
        if bm25_ok and dense_ok and hybrid_ok:
            ranks = {
                method: first_relevant_rank(runs[method].get(query_id, []), relevant, top_k)
                for method in METHODS
            }
            if len(set(ranks.values())) > 1:
                categories["All methods succeed with different ranks"].append(query_id)
    return dict(categories)


def format_query(
    query_id: str,
    queries_by_id: dict[str, dict],
    docs_by_id: dict[str, dict],
    qrels: dict[str, dict[str, int]],
    runs: dict[str, dict[str, list[dict]]],
) -> list[str]:
    query = queries_by_id[query_id]
    lines = [
        f"## Query: {query_id}",
        "",
        "Query text:",
        "",
        query["text"],
        "",
        "Relevant documents:",
    ]
    for doc_id in qrels[query_id]:
        doc = docs_by_id.get(doc_id, {"title": "", "text": ""})
        lines.append(f"- {doc_id} | {doc.get('title', '')}")
        lines.append(f"  {snippet(retrieval_text(doc))}")

    for method in METHODS:
        lines.extend(["", f"{method.upper()} top 5:"])
        for item in runs[method].get(query_id, [])[:5]:
            doc_id = str(item["doc_id"])
            doc = docs_by_id.get(doc_id, {"title": "", "text": ""})
            relevant = "yes" if doc_id in qrels[query_id] else "no"
            lines.append(
                f"{item['rank']}. {doc_id} | score={float(item['score']):.4f} | relevant={relevant} | {doc.get('title', '')}"
            )
            lines.append(f"   {snippet(retrieval_text(doc))}")
    lines.append("")
    return lines


def run(dataset: str, top_k: int, examples_per_category: int) -> None:
    docs, queries, qrels_rows = load_processed_dataset(dataset)
    docs_by_id = {str(doc["doc_id"]): doc for doc in docs}
    queries_by_id = {str(query["query_id"]): query for query in queries}
    qrels = qrels_by_query(qrels_rows)
    runs = load_runs(dataset)
    categories = categorise(runs, qrels, top_k)

    lines = [
        f"# {dataset} Failure Report",
        "",
        f"Top-k threshold: {top_k}",
        "",
        "This report samples cases where BM25, dense, and hybrid retrieval differ.",
        "",
    ]
    for category, query_ids in categories.items():
        lines.extend([f"# {category}", "", f"Available examples: {len(query_ids)}", ""])
        for query_id in query_ids[:examples_per_category]:
            lines.extend(format_query(query_id, queries_by_id, docs_by_id, qrels, runs))

    output = runs_dir(dataset) / "failure_report.md"
    output.write_text("\n".join(lines), encoding="utf-8")
    print(f"Wrote {output}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", default="scifact")
    parser.add_argument("--top-k", type=int, default=10)
    parser.add_argument("--examples-per-category", type=int, default=2)
    args = parser.parse_args()
    run(args.dataset, args.top_k, args.examples_per_category)


if __name__ == "__main__":
    main()
