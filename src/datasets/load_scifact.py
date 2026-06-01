from __future__ import annotations

import argparse
import csv
import json
import zipfile
from pathlib import Path

import requests

from src.common.io import ensure_dir, processed_dir, raw_dir, retrieval_text, write_jsonl
from src.common.text import tokenize


SCIFACT_URL = "https://public.ukp.informatik.tu-darmstadt.de/thakur/BEIR/datasets/scifact.zip"


def download(url: str, destination: Path) -> None:
    ensure_dir(destination.parent)
    with requests.get(url, stream=True, timeout=60) as response:
        response.raise_for_status()
        with destination.open("wb") as handle:
            for chunk in response.iter_content(chunk_size=1024 * 1024):
                if chunk:
                    handle.write(chunk)


def find_file(base: Path, name: str) -> Path:
    matches = list(base.rglob(name))
    if not matches:
        raise FileNotFoundError(f"Could not find {name} under {base}")
    return matches[0]


def normalise_corpus(path: Path) -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            raw = json.loads(line)
            rows.append(
                {
                    "doc_id": str(raw.get("_id", raw.get("doc_id"))),
                    "title": raw.get("title") or "",
                    "text": raw.get("text") or "",
                }
            )
    return rows


def normalise_queries(path: Path) -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            raw = json.loads(line)
            rows.append(
                {
                    "query_id": str(raw.get("_id", raw.get("query_id"))),
                    "text": raw.get("text") or "",
                }
            )
    return rows


def normalise_qrels(path: Path) -> list[dict[str, str | int]]:
    rows: list[dict[str, str | int]] = []
    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle, delimiter="\t")
        for raw in reader:
            query_id = raw.get("query-id") or raw.get("query_id")
            doc_id = raw.get("corpus-id") or raw.get("doc_id") or raw.get("corpus_id")
            score = raw.get("score", "1")
            if query_id is None or doc_id is None:
                continue
            rows.append({"query_id": str(query_id), "doc_id": str(doc_id), "score": int(score)})
    return rows


def prepare(force: bool = False) -> None:
    raw_base = raw_dir("scifact")
    output_base = processed_dir("scifact")
    corpus_out = output_base / "corpus.jsonl"
    queries_out = output_base / "queries.jsonl"
    qrels_out = output_base / "qrels.jsonl"

    if not force and corpus_out.exists() and queries_out.exists() and qrels_out.exists():
        print("Processed SciFact files already exist. Use --force to rebuild.")
        return

    archive = raw_base / "scifact.zip"
    extracted = raw_base / "extracted"

    if force and extracted.exists():
        for child in sorted(extracted.rglob("*"), reverse=True):
            if child.is_file():
                child.unlink()
            elif child.is_dir():
                child.rmdir()

    if not archive.exists():
        print(f"Downloading SciFact from {SCIFACT_URL}")
        download(SCIFACT_URL, archive)

    ensure_dir(extracted)
    if not any(extracted.iterdir()):
        with zipfile.ZipFile(archive) as zf:
            zf.extractall(extracted)

    corpus = normalise_corpus(find_file(extracted, "corpus.jsonl"))
    queries = normalise_queries(find_file(extracted, "queries.jsonl"))
    qrels = normalise_qrels(find_file(extracted, "test.tsv"))

    write_jsonl(corpus_out, corpus)
    write_jsonl(queries_out, queries)
    write_jsonl(qrels_out, qrels)

    doc_by_id = {row["doc_id"]: row for row in corpus}
    qrels_by_query: dict[str, list[str]] = {}
    for row in qrels:
        qrels_by_query.setdefault(str(row["query_id"]), []).append(str(row["doc_id"]))

    avg_doc_len = sum(len(tokenize(retrieval_text(row))) for row in corpus) / max(len(corpus), 1)
    avg_query_len = sum(len(tokenize(row["text"])) for row in queries) / max(len(queries), 1)

    print(f"documents: {len(corpus)}")
    print(f"queries: {len(queries)}")
    print(f"qrels: {len(qrels)}")
    print(f"average_document_tokens: {avg_doc_len:.1f}")
    print(f"average_query_tokens: {avg_query_len:.1f}")

    sample_query = next((query for query in queries if query["query_id"] in qrels_by_query), None)
    if sample_query:
        print()
        print(f"sample_query_id: {sample_query['query_id']}")
        print(f"sample_query: {sample_query['text']}")
        print("sample_relevant_titles:")
        for doc_id in qrels_by_query[sample_query["query_id"]][:3]:
            print(f"- {doc_id}: {doc_by_id.get(doc_id, {}).get('title', '')}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--force", action="store_true", help="Rebuild processed files")
    args = parser.parse_args()
    prepare(force=args.force)


if __name__ == "__main__":
    main()
