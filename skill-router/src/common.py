from __future__ import annotations

import os
from pathlib import Path

import torch
import torch.nn.functional as F
from transformers import AutoModel, AutoModelForCausalLM, AutoTokenizer


QUERY_INSTRUCTION = (
    "Instruct: Given a coding task description, retrieve the most relevant "
    "skill document that would help an agent complete the task\nQuery:"
)

RERANK_INSTRUCTION = (
    "Given a coding task description, judge whether the skill document "
    "is relevant and useful for completing the task"
)


def ensure_dir(path: str | Path) -> Path:
    p = Path(path)
    p.mkdir(parents=True, exist_ok=True)
    return p


def get_device() -> torch.device:
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def last_token_pool(last_hidden_states: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
    left_padding = (attention_mask[:, -1].sum() == attention_mask.shape[0])
    if left_padding:
        return last_hidden_states[:, -1]
    sequence_lengths = attention_mask.sum(dim=1) - 1
    batch_size = last_hidden_states.shape[0]
    return last_hidden_states[
        torch.arange(batch_size, device=last_hidden_states.device),
        sequence_lengths,
    ]


def format_query(raw_query: str, max_len: int = 1500) -> str:
    return f"{QUERY_INSTRUCTION}{raw_query[:max_len]}"


def format_skill(skill: dict, desc_max: int = 300, body_max: int = 2500) -> str:
    name = skill.get("name", "")
    desc = (skill.get("description") or "")[:desc_max]
    body = (skill.get("body") or "")[:body_max]
    return f"{name} | {desc} | {body}"


def format_rerank_prompt(
    name: str,
    desc: str,
    body: str,
    query_text: str,
    prompt_format: str = "flat-full",
    desc_max: int = 500,
    body_max: int = 2000,
) -> str:
    desc = desc[:desc_max]
    body = body[:body_max]

    if prompt_format == "flat-nd":
        doc_text = f"{name} | {desc}"
    elif prompt_format == "flat-full":
        doc_text = f"{name} | {desc} | {body}"
    elif prompt_format == "struct":
        return (
            f"<Instruct>: {RERANK_INSTRUCTION}\n\n"
            f"<Query>: {query_text}\n\n"
            f"<Skill>:\n"
            f"<Name>: {name}\n"
            f"<Description>: {desc}\n"
            f"<Body>: {body}"
        )
    else:
        raise ValueError(f"Unknown prompt_format: {prompt_format}")

    return (
        f"<Instruct>: {RERANK_INSTRUCTION}\n\n"
        f"<Query>: {query_text}\n\n"
        f"<Document>: {doc_text}"
    )


def load_embedding_model(model_name_or_path: str, dtype: torch.dtype = torch.bfloat16):
    tokenizer = AutoTokenizer.from_pretrained(
        model_name_or_path,
        trust_remote_code=True,
        padding_side="left",
    )
    model = AutoModel.from_pretrained(
        model_name_or_path,
        trust_remote_code=True,
        torch_dtype=dtype,
    )
    if tokenizer.pad_token is None and tokenizer.eos_token is not None:
        tokenizer.pad_token = tokenizer.eos_token
    return model, tokenizer


def load_reranker_model(model_name_or_path: str, dtype: torch.dtype = torch.bfloat16):
    tokenizer = AutoTokenizer.from_pretrained(
        model_name_or_path,
        padding_side="left",
        trust_remote_code=True,
    )
    model = AutoModelForCausalLM.from_pretrained(
        model_name_or_path,
        torch_dtype=dtype,
        trust_remote_code=True,
    )
    if tokenizer.pad_token is None and tokenizer.eos_token is not None:
        tokenizer.pad_token = tokenizer.eos_token
    return model, tokenizer


def encode_texts(model, tokenizer, texts: list[str], max_length: int, batch_size: int, device: torch.device) -> torch.Tensor:
    model.eval()
    all_embs: list[torch.Tensor] = []
    for i in range(0, len(texts), batch_size):
        batch_texts = texts[i:i + batch_size]
        encoded = tokenizer(
            batch_texts,
            padding=True,
            truncation=True,
            max_length=max_length,
            return_tensors="pt",
        )
        encoded = {k: v.to(device) for k, v in encoded.items()}
        with torch.no_grad():
            outputs = model(**encoded)
            embs = last_token_pool(outputs.last_hidden_state, encoded["attention_mask"])
            embs = F.normalize(embs, p=2, dim=1)
        all_embs.append(embs.cpu())
    return torch.cat(all_embs, dim=0)


def get_reranker_template_tokens(tokenizer):
    prefix = (
        '<|im_start|>system\nJudge whether the Document meets the requirements '
        'based on the Query and the Instruct provided. Note that the answer can '
        'only be "yes" or "no".<|im_end|>\n<|im_start|>user\n'
    )
    suffix = '<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n'
    prefix_tokens = tokenizer.encode(prefix, add_special_tokens=False)
    suffix_tokens = tokenizer.encode(suffix, add_special_tokens=False)
    return prefix_tokens, suffix_tokens


def tokenize_reranker_text(text: str, tokenizer, prefix_tokens, suffix_tokens, max_length: int) -> list[int]:
    inputs = tokenizer(
        text,
        padding=False,
        truncation=True,
        max_length=max_length - len(prefix_tokens) - len(suffix_tokens),
        return_attention_mask=False,
    )
    return prefix_tokens + inputs["input_ids"] + suffix_tokens
