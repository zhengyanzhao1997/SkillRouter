#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT_DIR"

ENCODER_MODEL_OR_PATH="${SKILLROUTER_EMB_MODEL_OR_PATH:-pipizhao/SkillRouter-Embedding-0.6B}"
RERANK_MODEL_OR_PATH="${SKILLROUTER_RERANK_MODEL_OR_PATH:-pipizhao/SkillRouter-Reranker-0.6B}"

python3 -m src.run_open_model_eval \
  --data_root "$ROOT_DIR/data/eval_core" \
  --encoder_model_or_path "$ENCODER_MODEL_OR_PATH" \
  --reranker_model_or_path "$RERANK_MODEL_OR_PATH" \
  --tiers easy hard \
  --retrieval_top_k 20 \
  --output_dir "$ROOT_DIR/outputs/open_model_eval" \
  "$@"
