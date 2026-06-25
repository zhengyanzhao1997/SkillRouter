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
        # Redirect print → stderr so model-loading logs don't corrupt
        # the JSON-RPC protocol (stdout = MCP data channel).
        old_stdout = sys.stdout
        sys.stdout = sys.stderr
        try:
            skills_dir = os.environ.get("SKILLS_DIR") or str(Path.home() / "skills_pool")
            _router = SkillRouter(skills_dir=skills_dir)
        finally:
            sys.stdout = old_stdout
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


if __name__ == "__main__":
    import anyio
    import anyio.lowlevel
    from contextlib import asynccontextmanager
    from mcp.types import JSONRPCMessage
    from mcp.server.session import SessionMessage

    @asynccontextmanager
    async def _stdio_transport():
        """Bypass mcp.stdio_server's TextIOWrapper bug on Python 3.13.

        Uses binary streams directly instead of wrapping with TextIOWrapper,
        which breaks under Python 3.13 when connected via subprocess PIPE.
        """
        stdin_bin = sys.stdin.buffer
        stdout_bin = sys.stdout.buffer

        async with anyio.wrap_file(stdin_bin) as stdin, \
                   anyio.wrap_file(stdout_bin) as stdout:
            read_writer, read_stream = anyio.create_memory_object_stream(0)
            write_stream, write_reader = anyio.create_memory_object_stream(0)

            async def _reader():
                try:
                    async with read_writer:
                        async for line in stdin:
                            line = line.decode("utf-8").strip()
                            if not line:
                                continue
                            try:
                                msg = JSONRPCMessage.model_validate_json(line)
                            except Exception as exc:
                                await read_writer.send(exc)
                                continue
                            await read_writer.send(SessionMessage(msg))
                except anyio.ClosedResourceError:
                    await anyio.lowlevel.checkpoint()

            async def _writer():
                try:
                    async with write_reader:
                        async for sm in write_reader:
                            data = sm.message.model_dump_json(
                                by_alias=True, exclude_none=True
                            )
                            await stdout.write((data + "\n").encode("utf-8"))
                            await stdout.flush()
                except anyio.ClosedResourceError:
                    await anyio.lowlevel.checkpoint()

            async with anyio.create_task_group() as tg:
                tg.start_soon(_reader)
                tg.start_soon(_writer)
                yield read_stream, write_stream

    async def _main():
        """Standalone entry: custom stdio transport + MCP server run loop."""
        async with _stdio_transport() as (read_stream, write_stream):
            await app._mcp_server.run(
                read_stream,
                write_stream,
                app._mcp_server.create_initialization_options(),
            )

    anyio.run(_main)
