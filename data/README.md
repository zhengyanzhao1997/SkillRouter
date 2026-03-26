# Data Layout

This directory contains the public evaluation benchmark for the SkillRouter release. The benchmark data is stored as GitHub-friendly `jsonl.gz` shards.

## Why Sharded `jsonl.gz`

- the raw JSONL files are larger than GitHub's per-file limit
- keeping the data as line-oriented shards preserves easy streaming
- the evaluation scripts in [../src/data_io.py](../src/data_io.py) can read either a single file or a shard directory

## Datasets

- [eval_core](eval_core/README.md)
