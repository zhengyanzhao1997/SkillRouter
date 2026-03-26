# Evaluation Protocol

This directory documents the public-release evaluation protocol.

## Benchmark Files

The benchmark data itself lives under [../data/eval_core/](../data/eval_core/).

- `tasks.jsonl`: benchmark tasks
- `relevance.json`: ground truth skill ids, graded relevance labels, and task type
- `easy/`, `hard/`: gzip-sharded skill pools

## Default Scoring

The default scored evaluation:

- start from all tasks in `tasks.jsonl`
- drop tasks where `task_type == "generic_only"`
- use the benchmark ground-truth set in `relevance.json`
- keep `relevance` as graded labels for nDCG

This yields a 75-task scored benchmark with 24 single-skill tasks and 51 multi-skill tasks.

Some benchmark task descriptions include execution-environment paths such as `/root/...` or `/home/...`. Those paths are part of the task specification, not a dependency on any maintainer-local machine.

## Prediction Format

Predictions are stored as a JSON object:

```json
{
  "task_id_1": ["skill_id_a", "skill_id_b", "skill_id_c"],
  "task_id_2": ["skill_id_x", "skill_id_y", "skill_id_z"]
}
```

An example file is included at [example_retrieval_submission.json](example_retrieval_submission.json).

## Metrics

[`../src/evaluate_predictions.py`](../src/evaluate_predictions.py) reports:

- `nDCG@1`, `nDCG@3`, `nDCG@10`
- `Hit@1`
- `Precision@3`
- `MRR@10`
- `Recall@10`, `Recall@20`, `Recall@50`
- `FullCoverage@3`, `FullCoverage@5`, `FullCoverage@10`

Results are aggregated over:

- `all`
- `single`
- `multi`

## Public Release Protocol

- evaluation tiers: `Easy + Hard`
- retrieval protocol: top-50 cosine candidates
- reranking protocol: retrieval top-20 followed by reranking

## Evaluate A Prediction File

```bash
bash scripts/evaluate_predictions.sh \
  --predictions outputs/custom_eval/retrieval/easy.json \
  --tier easy
```

## End-To-End Pipeline Evaluation

To evaluate the public 0.6B models end to end:

```bash
bash scripts/evaluate_open_models.sh --tiers easy hard
```
