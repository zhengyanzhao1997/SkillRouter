#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT_DIR"

python3 -m src.evaluate_predictions \
  --data_root "$ROOT_DIR/data/eval_core" \
  "$@"
