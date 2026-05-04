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
            "args": ["run", "-n", "dl", "python3", "/path/to/skill-router/src/mcp_server.py"],
            "env": {
                "SKILLS_DIR": "/path/to/skills_pool",
                "HF_HOME": "/path/to/huggingface_cache",
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

# Ensure router.py (same directory) is importable
sys.path.insert(0, str(Path(__file__).resolve().parent))

from router import SkillRouter  # noqa: E402

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
        "tasks. Two-step usage:\n"
        "1. Call search_skills to find relevant skills (returns summaries only).\n"
        "2. Call open_skill with the chosen skill's name to load its full content.\n"
        "Skip calling entirely for simple tasks (typo fixes, git operations, "
        "file I/O, general coding) that general knowledge can handle."
    ),
)


@app.tool(
    name="search_skills",
    description=(
        "Search for relevant specialized skills from a large skill pool. "
        "Returns top-3 skill summaries with name, description, and relevance score. "
        "Does NOT include full skill content — call open_skill after choosing one. "
        "Use this when the task involves research, plotting, paper writing, "
        "specific frameworks, experiment design, or domain-specific knowledge. "
        "Skip for simple tasks (typos, git, file I/O) — use general knowledge instead."
    ),
)
async def search_skills(task_description: str) -> str:
    """Route a task description to the most relevant skills (summaries only).

    Args:
        task_description: The user's task or question that needs a specialized skill.

    Returns:
        Markdown-formatted list of top-3 relevant skills (name + description only).
        Use open_skill to load the full content of a chosen skill.
    """
    router = get_router()
    results = router.search(task_description, top_k=3, include_body=False)

    if not results:
        return "No relevant skills found."

    lines = ["## 相关技能\n"]
    for i, skill in enumerate(results, 1):
        lines.append(f"### {i}. {skill['name']} (相关性: {skill['rerank_score']:.3f})")
        lines.append(f"**描述**: {skill['description']}")
        lines.append(f"**检索分数**: {skill['retrieval_score']}")
        lines.append("")
    lines.append("---")
    lines.append("使用 `open_skill` 加载某个技能的完整内容。")

    return "\n".join(lines)


@app.tool(
    name="open_skill",
    description=(
        "Load the full content of a specific skill by name. "
        "Call this after search_skills to get the complete instructions "
        "for the skill you want to use. The skill name should match "
        "exactly what search_skills returned."
    ),
)
async def open_skill(name: str) -> str:
    """Load the full content of a skill by name.

    Args:
        name: The exact skill name as returned by search_skills.

    Returns:
        Full skill content with name, description, and complete body.
    """
    router = get_router()
    skill = router.get_skill(name)

    if skill is None:
        available = [s["name"] for s in router.skills]
        return f"Skill '{name}' not found. Available skills: {', '.join(available[:10])}..."

    lines = [
        f"# {skill['name']}",
        f"**描述**: {skill['description']}",
        f"**内容**:\n```\n{skill['body']}\n```",
    ]
    return "\n".join(lines)
