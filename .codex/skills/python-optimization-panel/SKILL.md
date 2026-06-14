---
name: python-optimization-panel
description: Use when the user asks for Python code review, optimization, refactoring, hardening, production-quality cleanup, security/performance analysis, or explicitly references the saved python_optimization_prompt.
---

# Python Optimization Panel

This skill preserves the repository's original `python_optimization_prompt.md` and an additional
optimization methods handout for NumPy-heavy scientific Python.

## Use

When strict review/refactor/hardening mode is requested, read:

- `references/python_optimization_prompt.md`

Then follow that prompt unless it conflicts with higher-priority system or developer instructions.

When the task involves performance work, NumPy vectorization, EEG/ISC pipelines, bootstrap,
permutation testing, or correlation-heavy code, also read:

- `references/optimization_methods_notes.md`

The original PDF handout is preserved at:

- `references/optimization_methods.pdf`

## Local Adaptation

- For ordinary implementation tasks in this repository, do not force the saved prompt's phase gate unless the user asks for review/optimization/hardening or explicitly asks to use the saved prompt.
- For code changes, preserve public APIs where possible and verify with `uv run ruff check .` and `uv run pytest` when applicable.
- For EEG/ML code, combine this skill with `eeg-ml-research` so performance work does not compromise scientific validity.
