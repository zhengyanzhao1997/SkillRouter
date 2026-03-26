# Evaluation Benchmark

This directory contains the public-release evaluation benchmark.

## Contents

- `tasks.jsonl`
- `relevance.json`
- `easy/`
- `hard/`
- `manifest.json`

## Benchmark Structure

- `tasks.jsonl` contains the task descriptions to be routed
- `relevance.json` contains ground truth skill ids and graded relevance
- each tier directory contains a different benchmark skill pool

## Default Scoring

The default scored evaluation:

- exclude tasks marked `generic_only`
- score against the benchmark ground-truth set in `relevance.json`
- use `relevance` for graded nDCG

This produces 75 scored tasks out of 87 total benchmark tasks, with 24 single-skill tasks and 51 multi-skill tasks.

The benchmark task text itself contains paths such as `/root/...` and `/home/...` because the benchmark assumes an isolated execution workspace. Those paths are part of the task specification, not a dependency on the maintainer's local machine.

## Public Release Protocol

- evaluation tiers: `Easy + Hard`
- retrieval protocol: top-50 cosine retrieval
- reranking protocol: top-20 candidate reranking

## Source Attribution

This public benchmark release is assembled from upstream open-source resources:

- Ground-truth task queries and ground-truth skills are derived from [benchflow-ai/skillsbench](https://github.com/benchflow-ai/skillsbench).
- The benchmark skill pool is derived from [majiayu000/claude-skill-registry](https://github.com/majiayu000/claude-skill-registry).

If you use this evaluation data in downstream work, please acknowledge both the SkillRouter paper and the original upstream repositories above.

## Evaluate A Prediction File

```bash
bash scripts/evaluate_predictions.sh \
  --predictions outputs/custom_eval/retrieval/easy.json \
  --tier easy
```
