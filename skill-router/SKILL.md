---
name: skill-router
description: Dynamic skill discovery and routing via Bi-Encoder + Cross-Encoder. Self-contained deployment under ~/.merged_skills/skill-router/.
---

# SkillRouter — Dynamic Skill Discovery

This skill provides on-demand skill retrieval through an MCP server. Instead of loading all specialized skills into the system prompt, it only fetches relevant ones when needed.

## Architecture

All program files are self-contained in this directory:

```
skill-router/
├── SKILL.md                          ← This file (skill description)
├── src/
│   ├── mcp_server.py                 ← MCP server entry point
│   ├── router.py                     ← Bi-Encoder + Cross-Encoder pipeline
│   └── common.py                     ← Shared utilities
```

- MCP server runs locally via `conda run -n dl python3 skill-router/src/mcp_server.py`
- Skill pool lives at `~/skills_pool/skills/` (75+ skills with SKILL.md, auto-JSON-generation)

## When to Call `search_skills`

**调用场景**（需要 specialized skill 的复杂/领域任务）：
- 文献调研、科研方法、论文写作
- 数据可视化、绘图
- 特定框架/库的使用（PyTorch, LaTeX 等）
- 学术写作、基金申请
- 专利分析、Prior art 检索
- 实验设计、结果分析
- 任何需要专业知识指导的任务

**跳过场景**（通用能力即可处理）：
- 简单的代码修改（改 typo、修 bug、重构）
- Git 操作（commit、push、merge）
- 文件读写操作
- 解释简单代码片段
- 一般的 shell 命令
- 对话交流

## How to Use

1. **当任务属于调用场景**，调用 `search_skills` 获取相关技能
2. **审阅返回的技能**——按相关性排序，附带完整的 instructions
3. **使用 Top-1 或 Top-2 的技能**指导你的工作
4. **如果返回的技能不相关或为空**，用通用能力处理

## Important Notes

- The returned skill content includes executable instructions — follow them
- If the user's task spans multiple domains, call `search_skills` once with both keywords
- Be specific in your query — the router analyzes the full skill content (description + body)
- The MCP server is already configured and ready to use
