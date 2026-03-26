from __future__ import annotations

import gzip
import json
from pathlib import Path
from typing import Iterable


def iter_jsonl_paths(path: str | Path) -> list[Path]:
    p = Path(path)
    if p.is_file() and (p.name.endswith(".jsonl") or p.name.endswith(".jsonl.gz")):
        return [p]
    if p.is_dir():
        return sorted(
            [
                x
                for x in p.iterdir()
                if x.is_file() and (x.name.endswith(".jsonl") or x.name.endswith(".jsonl.gz"))
            ]
        )
    raise FileNotFoundError(f"Path not found: {p}")


def open_text(path: str | Path):
    p = Path(path)
    if p.name.endswith(".gz"):
        return gzip.open(p, "rt", encoding="utf-8")
    return p.open("r", encoding="utf-8")


def load_jsonl(path: str | Path) -> list[dict]:
    records: list[dict] = []
    for file_path in iter_jsonl_paths(path):
        with open_text(file_path) as f:
            for line in f:
                if line.strip():
                    records.append(json.loads(line))
    return records


def stream_jsonl(path: str | Path) -> Iterable[dict]:
    for file_path in iter_jsonl_paths(path):
        with open_text(file_path) as f:
            for line in f:
                if line.strip():
                    yield json.loads(line)


def count_jsonl(path: str | Path) -> int:
    count = 0
    for file_path in iter_jsonl_paths(path):
        with open_text(file_path) as f:
            for line in f:
                if line.strip():
                    count += 1
    return count
