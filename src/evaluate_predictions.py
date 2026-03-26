from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from src.data_io import load_jsonl
from src.metrics import compute_all_metrics


TIER_FILES = {
    "easy": "easy",
    "hard": "hard",
}


def aggregate(metrics_list: list[dict]) -> dict:
    if not metrics_list:
        return {}
    out = {}
    for key in metrics_list[0]:
        out[key] = float(np.mean([m[key] for m in metrics_list]))
    out["count"] = len(metrics_list)
    return out


def main():
    parser = argparse.ArgumentParser(description="Evaluate retrieval or reranked predictions on SkillRouter Eval Core.")
    parser.add_argument("--data_root", required=True)
    parser.add_argument("--predictions", required=True)
    parser.add_argument("--task_mode", choices=["core", "all", "single"], default="core")
    parser.add_argument("--tier", choices=["easy", "hard"], required=True)
    parser.add_argument("--output_json", default="")
    args = parser.parse_args()

    data_root = Path(args.data_root)
    tier_pool = load_jsonl(data_root / TIER_FILES[args.tier])
    pool_id_set = {x["skill_id"] for x in tier_pool}
    tasks = load_jsonl(data_root / "tasks.jsonl")
    relevance = json.loads((data_root / "relevance.json").read_text())
    predictions = json.loads(Path(args.predictions).read_text())

    results_by_stratum = {"all": [], "single": [], "multi": []}
    for task in tasks:
        task_id = task["task_id"]
        rel_entry = relevance.get(task_id, {})
        task_type = rel_entry.get("task_type")
        if args.task_mode == "core":
            if task_type == "generic_only":
                continue
            gt_ids = set(rel_entry.get("core_gt_ids", rel_entry.get("gt_skill_ids", [])))
        elif args.task_mode == "single":
            gt_ids = set(rel_entry.get("gt_skill_ids", []))
            if len(gt_ids) != 1:
                continue
        else:
            gt_ids = set(rel_entry.get("gt_skill_ids", []))

        gt_ids_in_pool = gt_ids & pool_id_set
        if not gt_ids_in_pool or task_id not in predictions:
            continue

        ranked_ids = predictions[task_id]
        tier_relevance = {
            k: float(v) for k, v in rel_entry.get("relevance", {}).items() if k in pool_id_set
        }
        metrics = compute_all_metrics(ranked_ids, gt_ids_in_pool, tier_relevance or None)
        results_by_stratum["all"].append(metrics)
        if len(gt_ids) == 1:
            results_by_stratum["single"].append(metrics)
        else:
            results_by_stratum["multi"].append(metrics)

    aggregated = {k: aggregate(v) for k, v in results_by_stratum.items() if v}
    text = json.dumps(aggregated, indent=2)
    print(text)
    if args.output_json:
        Path(args.output_json).write_text(text)


if __name__ == "__main__":
    main()
