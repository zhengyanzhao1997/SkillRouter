#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "$0")/.." && pwd)"
DATA_REPO="${SKILLROUTER_DATA_REPO:-pipizhao/SkillRouter-Eval-Core}"
DATA_DIR="${SKILLROUTER_DATA_DIR:-$ROOT_DIR/data/eval_core}"

python3 - "$DATA_REPO" "$DATA_DIR" <<'PY'
from __future__ import annotations

import sys
from pathlib import Path

try:
    from huggingface_hub import snapshot_download
except ImportError as exc:
    raise SystemExit(
        "Missing huggingface_hub. Run `pip install -r requirements.txt` "
        "or `pip install huggingface_hub`, then retry."
    ) from exc

repo_id, local_dir = sys.argv[1], sys.argv[2]
Path(local_dir).mkdir(parents=True, exist_ok=True)

snapshot_download(
    repo_id=repo_id,
    repo_type="dataset",
    local_dir=local_dir,
    allow_patterns=[
        "manifest.json",
        "tasks.jsonl",
        "relevance.json",
        "easy/*.jsonl.gz",
        "hard/*.jsonl.gz",
    ],
)

print(f"Downloaded {repo_id} to {local_dir}")
PY
