from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

from tqdm import tqdm

from src.common.io import load_processed_dataset, read_json, read_jsonl, runs_dir, write_json, write_jsonl
from src.eval.metrics import compute_metrics
from src.retrievers.bm25 import BM25Retriever
from src.retrievers.dense import DenseRetriever
from src.retrievers.hybrid import HybridRetriever


def build_retriever(name: str, dataset: str, docs: list[dict]) -> Any:
    if name == "bm25":
        return BM25Retriever.build(docs)
    if name == "dense":
        return DenseRetriever.build(dataset, docs)
    if name == "hybrid":
        return HybridRetriever.build(dataset, docs)
    raise ValueError(f"Unknown retriever: {name}")


def timing_dict(retriever: Any) -> dict:
    if hasattr(retriever, "timing_dict"):
        return retriever.timing_dict()
    return retriever.timing.to_dict()


def run(dataset: str, retriever_name: str, top_k: int) -> None:
    docs, queries, qrels = load_processed_dataset(dataset)
    retriever = build_retriever(retriever_name, dataset, docs)

    rows = []
    for query in tqdm(queries, desc=f"{retriever_name} retrieval"):
        rows.append(
            {
                "query_id": str(query["query_id"]),
                "retriever": retriever_name,
                "results": retriever.search(query["text"], top_k),
            }
        )

    output_dir = runs_dir(dataset) / retriever_name
    write_jsonl(output_dir / "results.jsonl", rows)

    metrics = compute_metrics(rows, qrels)
    timing = timing_dict(retriever)
    write_json(output_dir / "metrics.json", metrics)
    write_json(output_dir / "timing.json", timing)
    update_summary(dataset)
    print_summary(retriever_name, metrics, timing)


def update_summary(dataset: str) -> None:
    base = runs_dir(dataset)
    methods = ["bm25", "dense", "hybrid"]
    summary: dict[str, dict[str, Any]] = {}
    for method in methods:
        metrics_path = base / method / "metrics.json"
        timing_path = base / method / "timing.json"
        if metrics_path.exists() and timing_path.exists():
            summary[method] = {
                "metrics": read_json(metrics_path),
                "timing": read_json(timing_path),
            }

    write_json(base / "summary.json", summary)
    write_summary_md(base / "summary.md", dataset, summary)


def fmt(value: Any, digits: int = 4) -> str:
    if isinstance(value, float):
        return f"{value:.{digits}f}"
    return str(value)


def write_summary_md(path: Path, dataset: str, summary: dict[str, dict[str, Any]]) -> None:
    lines = [
        f"# {dataset} Retrieval Summary",
        "",
        "Local benchmark timings are relative measurements for this implementation, not production latency claims.",
        "",
        "| method | recall@5 | recall@10 | recall@100 | mrr@10 | ndcg@10 | mean ms | p95 ms |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for method, payload in summary.items():
        metrics = payload["metrics"]
        timing = payload["timing"]
        lines.append(
            "| "
            + " | ".join(
                [
                    method,
                    fmt(metrics.get("recall@5", 0.0)),
                    fmt(metrics.get("recall@10", 0.0)),
                    fmt(metrics.get("recall@100", 0.0)),
                    fmt(metrics.get("mrr@10", 0.0)),
                    fmt(metrics.get("ndcg@10", 0.0)),
                    fmt(timing.get("query_latency_mean_ms", 0.0), 2),
                    fmt(timing.get("query_latency_p95_ms", 0.0), 2),
                ]
            )
            + " |"
        )
    lines.extend(["", "## Notes", "", "- Dense retrieval uses `sentence-transformers/all-MiniLM-L6-v2`.", "- Hybrid retrieval uses Reciprocal Rank Fusion with `k = 60`."])
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def print_summary(retriever_name: str, metrics: dict, timing: dict) -> None:
    print(f"{retriever_name} metrics:")
    for key, value in metrics.items():
        print(f"  {key}: {value:.4f}")
    print("timing:")
    for key in ("query_latency_mean_ms", "query_latency_p50_ms", "query_latency_p95_ms", "query_latency_max_ms"):
        print(f"  {key}: {timing.get(key, 0.0):.2f}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", default="scifact")
    parser.add_argument("--retriever", choices=["bm25", "dense", "hybrid"], required=True)
    parser.add_argument("--top-k", type=int, default=100)
    args = parser.parse_args()
    run(args.dataset, args.retriever, args.top_k)


if __name__ == "__main__":
    main()
