"""SkillRouter: Bi-Encoder retrieval + Cross-Encoder reranking for skill routing."""

from __future__ import annotations

import hashlib
import json
import os
import time
from pathlib import Path

import yaml
import torch
import torch.nn.functional as F

from src.common import (
    encode_texts,
    format_query,
    format_rerank_prompt,
    format_skill,
    get_device,
    get_reranker_template_tokens,
    load_embedding_model,
    load_reranker_model,
    tokenize_reranker_text,
)


def _snapshot_path(model_type: str) -> str:
    """Resolve cache path for a SkillRouter model snapshot."""
    model_type_lower = model_type.lower()
    cache = Path.home() / ".cache" / "skillrouter" / model_type_lower
    snapshots_dir = cache / f"models--pipizhao--SkillRouter-{model_type}-0.6B" / "snapshots"
    if not snapshots_dir.is_dir():
        return f"pipizhao/SkillRouter-{model_type}-0.6B"
    snapshots = [p for p in snapshots_dir.iterdir() if p.is_dir()]
    if not snapshots:
        return f"pipizhao/SkillRouter-{model_type}-0.6B"
    return str(snapshots[0])


class SkillRouter:
    """Compact full-text retrieve-and-rerank pipeline for skill routing."""

    def __init__(
        self,
        skills_dir: str | Path | None = None,
        device: str | torch.device | None = None,
        emb_model_or_path: str | None = None,
        rerank_model_or_path: str | None = None,
        retrieval_top_k: int = 20,
        final_top_k: int = 3,
        encoder_max_length: int = 4096,
        reranker_max_length: int = 4096,
        encoder_batch_size: int = 8,
        reranker_batch_size: int = 8,
        desc_max: int = 500,
        body_max: int = 2000,
        encoder_body_max: int = 4096,
        cache_dir: str | Path | None = None,
    ):
        self.device = get_device() if device is None else torch.device(device)
        self.retrieval_top_k = retrieval_top_k
        self.final_top_k = final_top_k
        self.encoder_max_length = encoder_max_length
        self.reranker_max_length = reranker_max_length
        self.encoder_batch_size = encoder_batch_size
        self.reranker_batch_size = reranker_batch_size
        self.desc_max = desc_max
        self.body_max = body_max
        self.encoder_body_max = encoder_body_max
        self.cache_dir = Path(cache_dir) if cache_dir else Path.home() / ".cache" / "skillrouter"

        # Load models
        emb_path = emb_model_or_path or _snapshot_path("Embedding")
        rerank_path = rerank_model_or_path or _snapshot_path("Reranker")

        print(f"[SkillRouter] Loading embedding model from: {emb_path}")
        self.emb_model, self.emb_tokenizer = load_embedding_model(emb_path)
        self.emb_model.to(self.device).eval()

        print(f"[SkillRouter] Loading reranker model from: {rerank_path}")
        self.rr_model, self.rr_tokenizer = load_reranker_model(rerank_path)
        self.rr_model.to(self.device).eval()

        # Pre-compute reranker template tokens
        self._prefix_tokens, self._suffix_tokens = get_reranker_template_tokens(self.rr_tokenizer)
        self._token_true_id = self.rr_tokenizer.convert_tokens_to_ids("yes")
        self._token_false_id = self.rr_tokenizer.convert_tokens_to_ids("no")

        # Load skills and pre-compute embeddings
        self.skills: list[dict] = []
        self.pool_ids: list[str] = []
        self.skill_embeddings: torch.Tensor | None = None

        skills_dir = Path(skills_dir or Path.home() / "skills_pool")
        self._load_and_index(skills_dir)

    def _parse_skill_md(self, md_path: Path) -> dict | None:
        """Parse a SKILL.md file into a skill dict with optional YAML frontmatter."""
        try:
            content = md_path.read_text(encoding="utf-8")
        except Exception:
            return None

        name = md_path.parent.stem
        description = ""
        body = content

        if content.startswith("---"):
            parts = content.split("---", 2)
            if len(parts) >= 3:
                try:
                    meta = yaml.safe_load(parts[1])
                    if isinstance(meta, dict):
                        name = meta.get("name") or name
                        description = meta.get("description") or ""
                    body = parts[2].strip()
                except Exception:
                    pass

        return {
            "skill_id": name,
            "name": name,
            "description": description,
            "body": body,
        }

    def _sync_skill_dirs(self, skills_dir: Path) -> list[Path]:
        """Scan for directories with SKILL.md and generate JSON files.

        Generates a <skill-name>.json inside each skill directory.
        Returns updated list of all JSON file paths (including auto-generated ones).
        """
        for md_path in sorted(skills_dir.rglob("SKILL.md")):
            # JSON goes inside the skill directory alongside SKILL.md
            json_path = md_path.parent / f"{md_path.parent.name}.json"
            # Skip if JSON exists and is newer than SKILL.md
            if json_path.exists() and json_path.stat().st_mtime >= md_path.stat().st_mtime:
                continue

            skill = self._parse_skill_md(md_path)
            if skill is None:
                continue

            json_path.write_text(
                json.dumps(skill, ensure_ascii=False, indent=2),
                encoding="utf-8",
            )
            print(f"[SkillRouter] Generated {json_path.relative_to(skills_dir)} from SKILL.md")

        return sorted(skills_dir.rglob("*.json"))

    def _cache_key(self, json_files: list[Path]) -> str:
        """Compute a cache key from file metadata and encoder config."""
        parts = [f"encoder_body_max={self.encoder_body_max}",
                 f"encoder_max_length={self.encoder_max_length}"]
        for fp in json_files:
            st = fp.stat()
            parts.append(f"{fp.name}:{st.st_mtime_ns}:{st.st_size}")
        return hashlib.sha256("|".join(parts).encode()).hexdigest()

    def _load_and_index(self, skills_dir: Path) -> None:
        """Load skills from directory tree and pre-compute embeddings (with caching)."""
        # Auto-detect skill dirs (SKILL.md) and generate JSONs
        json_files = self._sync_skill_dirs(skills_dir)

        if not json_files:
            raise FileNotFoundError(f"No skill JSON files found in {skills_dir}")

        skills = []
        seen: set[str] = set()
        for fp in json_files:
            data = json.loads(fp.read_text(encoding="utf-8"))
            name = data.get("name", fp.stem)
            if name in seen:
                continue  # deduplicate: first file wins (sorted by path)
            seen.add(name)
            skills.append({
                "skill_id": data.get("skill_id", name),
                "name": name,
                "description": data.get("description") or "",
                "body": data.get("body") or "",
            })

        self.skills = skills
        self.pool_ids = [s["skill_id"] for s in skills]

        # Try loading cached embeddings
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        cache_meta_path = self.cache_dir / "embedding_cache.json"
        cache_emb_path = self.cache_dir / "skill_embeddings.pt"
        current_key = self._cache_key(json_files)

        if cache_meta_path.exists() and cache_emb_path.exists():
            try:
                meta = json.loads(cache_meta_path.read_text())
                if meta.get("cache_key") == current_key and meta.get("n_skills") == len(skills):
                    self.skill_embeddings = torch.load(cache_emb_path, weights_only=False)
                    print(f"[SkillRouter] Loaded {len(skills)} cached embeddings from {cache_emb_path}")
                    return
            except Exception:
                pass  # Fall through to re-encode

        # Cache miss — encode and save
        pool_texts = [
            format_skill(s, desc_max=self.desc_max, body_max=self.encoder_body_max)
            for s in skills
        ]
        print(f"[SkillRouter] Encoding {len(pool_texts)} skills...")
        t0 = time.time()
        self.skill_embeddings = encode_texts(
            self.emb_model, self.emb_tokenizer, pool_texts,
            self.encoder_max_length, self.encoder_batch_size, self.device,
        )
        elapsed = time.time() - t0
        print(f"[SkillRouter] Encoded {len(pool_texts)} skills in {elapsed:.2f}s "
              f"({len(pool_texts)/elapsed:.0f} skills/s)")

        # Save cache (atomically via temp file)
        tmp_meta = cache_meta_path.with_suffix(".tmp.json")
        tmp_emb = cache_emb_path.with_suffix(".tmp.pt")
        try:
            json.dump({"cache_key": current_key, "n_skills": len(skills),
                       "created_at": time.time()}, tmp_meta.open("w"))
            torch.save(self.skill_embeddings.cpu(), tmp_emb)
            tmp_meta.replace(cache_meta_path)
            tmp_emb.replace(cache_emb_path)
            print(f"[SkillRouter] Cached embeddings to {cache_emb_path}")
        except Exception as e:
            print(f"[SkillRouter] Warning: failed to cache embeddings: {e}")
            for p in [tmp_meta, tmp_emb]:
                p.unlink(missing_ok=True)

    @torch.no_grad()
    def search(self, query: str, top_k: int | None = None) -> list[dict]:
        """Route a query to the most relevant skills.

        Returns top_k skills with their name, description, body, and scores.
        """
        k = top_k or self.final_top_k
        if self.skill_embeddings is None:
            raise RuntimeError("No skills indexed — call _load_and_index first")

        # --- Stage 1: Bi-Encoder retrieval ---
        query_text = format_query(query, max_len=2000)
        q_emb = encode_texts(
            self.emb_model, self.emb_tokenizer, [query_text],
            self.encoder_max_length, self.encoder_batch_size, self.device,
        )
        sims = (q_emb @ self.skill_embeddings.T).squeeze(0)
        topk_scores, topk_idx = torch.topk(sims, min(self.retrieval_top_k, len(self.skills)))
        topk_scores = topk_scores.tolist()
        topk_idx = topk_idx.tolist()

        candidates = [self.skills[i] for i in topk_idx]

        # --- Stage 2: Cross-Encoder reranking ---
        texts = [
            format_rerank_prompt(
                c["name"], c["description"], c["body"], query,
                prompt_format="flat-full",
            )
            for c in candidates
        ]
        tokenized = [
            tokenize_reranker_text(t, self.rr_tokenizer, self._prefix_tokens,
                                   self._suffix_tokens, self.reranker_max_length)
            for t in texts
        ]

        scores: list[float] = []
        pad_id = self.rr_tokenizer.pad_token_id or 0
        for i in range(0, len(tokenized), self.reranker_batch_size):
            batch_ids = tokenized[i:i + self.reranker_batch_size]
            max_len = max(len(x) for x in batch_ids)
            padded, masks = [], []
            for ids in batch_ids:
                pad_len = max_len - len(ids)
                padded.append([pad_id] * pad_len + ids)
                masks.append([0] * pad_len + [1] * len(ids))
            input_ids = torch.tensor(padded, dtype=torch.long, device=self.device)
            attention_mask = torch.tensor(masks, dtype=torch.long, device=self.device)
            logits = self.rr_model(input_ids=input_ids, attention_mask=attention_mask).logits[:, -1, :]
            batch_scores = (logits[:, self._token_true_id] - logits[:, self._token_false_id]).float().cpu().tolist()
            scores.extend(batch_scores)

        # Combine and rank by reranker score
        ranked = sorted(
            [
                (candidates[j], scores[j], topk_scores[j])
                for j in range(len(candidates))
            ],
            key=lambda x: x[1],
            reverse=True,
        )

        results = []
        for skill, rerank_score, retrieval_score in ranked[:k]:
            results.append({
                "skill_id": skill["skill_id"],
                "name": skill["name"],
                "description": skill["description"],
                "body": skill["body"],
                "retrieval_score": round(retrieval_score, 4),
                "rerank_score": round(rerank_score, 4),
            })
        return results


    def search_retrieval_only(self, query: str, top_k: int = 20) -> list[dict]:
        """Stage-1 only: Bi-Encoder retrieval without reranking."""
        query_text = format_query(query, max_len=2000)
        q_emb = encode_texts(
            self.emb_model, self.emb_tokenizer, [query_text],
            self.encoder_max_length, self.encoder_batch_size, self.device,
        )
        sims = (q_emb @ self.skill_embeddings.T).squeeze(0)
        topk_scores, topk_idx = torch.topk(sims, min(top_k, len(self.skills)))

        results = []
        for score, idx in zip(topk_scores.tolist(), topk_idx.tolist()):
            s = self.skills[idx]
            results.append({
                "skill_id": s["skill_id"],
                "name": s["name"],
                "description": s["description"],
                "retrieval_score": round(score, 4),
            })
        return results
