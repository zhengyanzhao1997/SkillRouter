#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT_DIR"
DATA_ROOT="$ROOT_DIR/data/eval_core"

for tier in easy hard; do
  if ! find "$DATA_ROOT/$tier" -name '*.jsonl.gz' -print -quit >/dev/null 2>&1; then
    echo "Missing SkillRouter eval data shards for tier: $tier" >&2
    echo "Run: bash scripts/download_eval_data.sh" >&2
    exit 1
  fi
done

python3 -m src.evaluate_predictions \
  --data_root "$DATA_ROOT" \
  "$@"
