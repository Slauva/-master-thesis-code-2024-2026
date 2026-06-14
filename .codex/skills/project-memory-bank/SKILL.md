---
name: project-memory-bank
description: Use when reading, updating, or relying on this repository's memory bank for project context, decisions, experiments, next actions, or durable research notes.
---

# Project Memory Bank

## Files

- `.codex/memory-bank/project_brief.md`: stable project purpose and baseline context.
- `.codex/memory-bank/active_context.md`: current focus, immediate next actions, and open questions.
- `.codex/memory-bank/decisions.md`: durable scientific and engineering decisions.
- `.codex/memory-bank/experiments.md`: experiment runs, metrics, observations, and failures.
- `.codex/memory-bank/glossary.md`: project-specific terms and abbreviations.

## Workflow

1. Before substantive work, read `project_brief.md`, `active_context.md`, and relevant sections of `decisions.md`.
2. During work, keep new facts separate from assumptions.
3. After meaningful progress, update the smallest relevant memory file.
4. Date durable decisions using `YYYY-MM-DD`.
5. For experiments, record enough detail to reproduce the result.

## Rules

- Do not store secrets, API keys, credentials, or sensitive raw personal data.
- Do not use memory files as a substitute for source code, tests, or experiment artifacts.
- Prefer concise notes over narrative logs.
- If a prior memory entry is wrong, append a correction with date instead of silently rewriting history, unless the user explicitly asks for cleanup.
