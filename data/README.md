# Data Layout

This directory contains the public evaluation benchmark metadata for the SkillRouter release. The large skill-pool shards are hosted on Hugging Face Datasets:

- [pipizhao/SkillRouter-Eval-Core](https://huggingface.co/datasets/pipizhao/SkillRouter-Eval-Core)

Download the full benchmark into `data/eval_core`:

```bash
bash scripts/download_eval_data.sh
```

The downloaded benchmark skill pools are stored as `jsonl.gz` shards under `data/eval_core/easy/` and `data/eval_core/hard/`.

## Why Sharded `jsonl.gz`

- the raw JSONL files are larger than GitHub's per-file limit
- keeping the data as line-oriented shards preserves easy streaming
- the evaluation scripts in [../src/data_io.py](../src/data_io.py) can read either a single file or a shard directory

## Datasets

- [eval_core](eval_core/README.md)
