"""
SkillRouter MCP Server

Exposes SkillRouter's bi-encoder + cross-encoder pipeline as MCP tools.
Claude Code connects via stdio transport.

Usage:
    python src/mcp_server.py

Configure in ~/.claude/settings.json:
{
    "mcpServers": {
        "skill-router": {
            "command": "conda",
            "args": ["run", "-n", "dl", "python3", "/path/to/src/mcp_server.py"],
            "env": {
                "SKILLS_DIR": "/home/xhkzdepartedream/skills_pool",
                "HF_HOME": "/home/xhkzdepartedream/.cache/huggingface",
                "HF_HUB_DISABLE_XET": "1"
            }
        }
    }
}
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path

from mcp.server.fastmcp import FastMCP

# Ensure the SkillRouter package is importable
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.router import SkillRouter  # noqa: E402

# Global router instance (lazy-loaded on first tool call)
_router: SkillRouter | None = None


def get_router() -> SkillRouter:
    global _router
    if _router is None:
        skills_dir = os.environ.get("SKILLS_DIR") or str(Path.home() / "skills_pool")
        _router = SkillRouter(skills_dir=skills_dir)
    return _router


# Create MCP server
app = FastMCP(
    "skill-router",
    instructions=(
        "SkillRouter provides dynamic skill discovery for complex or domain-specific "
        "tasks. Call search_skills when the task involves research, plotting, paper "
        "writing, specific frameworks (PyTorch, LaTeX, etc.), experiment design, or "
        "any area that may need specialized knowledge. Skip calling for simple tasks "
        "(typo fixes, git operations, file I/O, general coding) that general knowledge "
        "can handle. The returned skills contain executable instructions — use the "
        "top-returned skill to guide your work."
    ),
)


@app.tool(
    name="search_skills",
    description=(
        "Search for relevant specialized skills from a large skill pool. "
        "Use this when the task involves research, plotting, paper writing, "
        "specific frameworks, experiment design, or domain-specific knowledge. "
        "Returns top-3 most relevant skills with full instructions. "
        "Skip for simple tasks (typos, git, file I/O) — use general knowledge instead."
    ),
)
async def search_skills(task_description: str) -> str:
    """Route a task description to the most relevant skills.

    Args:
        task_description: The user's task or question that needs a specialized skill.

    Returns:
        Markdown-formatted list of top-3 relevant skills with full content.
    """
    router = get_router()
    results = router.search(task_description, top_k=3)

    if not results:
        return "No relevant skills found."

    lines = ["## 相关技能\n"]
    for i, skill in enumerate(results, 1):
        lines.append(f"### {i}. {skill['name']} (相关性: {skill['rerank_score']:.3f})")
        lines.append(f"**描述**: {skill['description']}")
        lines.append(f"**内容**:\n```\n{skill['body']}\n```\n")

    return "\n".join(lines)


if __name__ == "__main__":
    app.run(transport="stdio")
