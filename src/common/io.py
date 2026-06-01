from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Iterable


ROOT = Path(__file__).resolve().parents[2]


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def write_jsonl(path: Path, rows: Iterable[dict[str, Any]]) -> None:
    ensure_dir(path.parent)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def read_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def write_json(path: Path, data: Any) -> None:
    ensure_dir(path.parent)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(data, handle, indent=2, ensure_ascii=False)
        handle.write("\n")


def processed_dir(dataset: str) -> Path:
    return ROOT / "data" / "processed" / dataset


def raw_dir(dataset: str) -> Path:
    return ROOT / "data" / "raw" / dataset


def runs_dir(dataset: str) -> Path:
    return ROOT / "runs" / dataset


def load_processed_dataset(dataset: str) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    base = processed_dir(dataset)
    return (
        read_jsonl(base / "corpus.jsonl"),
        read_jsonl(base / "queries.jsonl"),
        read_jsonl(base / "qrels.jsonl"),
    )


def retrieval_text(doc: dict[str, Any]) -> str:
    title = doc.get("title") or ""
    text = doc.get("text") or ""
    return f"{title}\n{text}".strip()
