from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from src.common import ensure_dir, format_query, format_skill, get_device, load_embedding_model, encode_texts
from src.data_io import load_jsonl


TIER_FILES = {
    "easy": "easy",
    "hard": "hard",
}


def main():
    parser = argparse.ArgumentParser(description="Export top-K retrieval results for SkillRouter benchmark tiers.")
    parser.add_argument("--encoder_model_or_path", required=True)
    parser.add_argument("--data_root", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--top_k", type=int, default=50)
    parser.add_argument("--tiers", nargs="+", choices=["easy", "hard"], default=["easy", "hard"])
    parser.add_argument("--max_length", type=int, default=4096)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--desc_max", type=int, default=500)
    parser.add_argument("--body_max", type=int, default=8000)
    args = parser.parse_args()

    data_root = Path(args.data_root)
    output_dir = ensure_dir(args.output_dir)
    retrieval_dir = ensure_dir(output_dir / "retrieval")

    device = get_device()
    model, tokenizer = load_embedding_model(args.encoder_model_or_path)
    model.to(device).eval()

    tasks = load_jsonl(data_root / "tasks.jsonl")
    query_texts = [format_query(t.get("instruction_text", t.get("query", "")), max_len=2000) for t in tasks]
    query_ids = [t["task_id"] for t in tasks]
    query_embs = encode_texts(model, tokenizer, query_texts, args.max_length, args.batch_size, device)

    for tier in args.tiers:
        tier_stem = TIER_FILES[tier]
        tier_pool = load_jsonl(data_root / tier_stem)
        pool_ids = [s.get("skill_id") or s.get("id") or "" for s in tier_pool]
        pool_texts = [format_skill(s, desc_max=args.desc_max, body_max=args.body_max) for s in tier_pool]
        pool_embs = encode_texts(model, tokenizer, pool_texts, args.max_length, args.batch_size, device)
        sim_matrix = query_embs @ pool_embs.T
        retrieval_results = {}
        for qi, task_id in enumerate(query_ids):
            sims = sim_matrix[qi]
            _, topk_idx = torch.topk(sims, min(args.top_k, len(pool_ids)))
            retrieval_results[task_id] = [pool_ids[idx] for idx in topk_idx.tolist()]
        out_path = retrieval_dir / f"{tier}.json"
        out_path.write_text(json.dumps(retrieval_results, indent=2, ensure_ascii=False))
        print(f"[saved] {out_path}")


if __name__ == "__main__":
    main()
