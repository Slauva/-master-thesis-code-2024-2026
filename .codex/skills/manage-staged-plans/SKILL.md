---
name: manage-staged-plans
description: Create and execute implementation or research plans as explicit reviewable stages with user approval gates, durable memory-bank tracking, verification evidence, and Jupyter visualization artifacts. Use when the user asks for a plan, phased implementation, checkpoints, approval after each stage, persistent progress tracking, or a substantial multi-stage project.
---

# Manage Staged Plans

Create plans that remain useful across turns. Separate planning, approval, execution, verification,
and memory updates instead of treating a plan as a disposable checklist.

## Startup

1. Read `.codex/memory-bank/project_brief.md`.
2. Read `.codex/memory-bank/active_context.md`.
3. Read relevant constraints in `.codex/memory-bank/decisions.md`.
4. Inspect only the source files needed to understand the requested work.
5. Use `project-memory-bank` whenever it is available.

Keep facts, assumptions, open questions, and proposed decisions visibly separate.

## Create The Plan

1. Define the goal, scope, exclusions, acceptance criteria, and major risks.
2. Split the work into ordered stages that each produce a coherent, reviewable result.
3. Give every stage:
   - a concrete objective;
   - intended files or artifacts;
   - scientific or engineering constraints;
   - verification commands or checks;
   - completion criteria;
   - a mandatory user review gate.
4. Put risky contracts, leakage boundaries, schemas, and architecture before dependent
   implementation work.
5. Keep stages small enough that rejecting or revising one stage does not invalidate unrelated
   completed work.
6. Present the complete draft plan to the user and request explicit approval.

Do not mark the plan approved, persist it as the active implementation plan, or begin stage 1 until
the user explicitly approves it. Treat requested changes as plan revisions and present the revised
plan again.

## Save The Approved Plan

After explicit approval:

1. Save the plan to `.codex/memory-bank/plans/YYYY-MM-DD-<short-slug>.md`.
2. Set:
   - `Status: approved`;
   - `Last updated: YYYY-MM-DD`;
   - `Next stage: 1 - <stage name>`.
3. Record each stage as `Pending`.
4. Add the plan path and current stage to `.codex/memory-bank/active_context.md`.
5. Mirror the stages into the available task-plan tool when useful.

Use this minimum plan shape:

```markdown
# <Plan Title>

Status: approved
Last updated: YYYY-MM-DD
Next stage: 1 - <Stage Name>

## Goal

## Scope

## Acceptance Criteria

## Stages

### 1. <Stage Name> - Pending

- Objective:
- Deliverables:
- Constraints:
- Verification:
- Completion criteria:
- Review gate: Stop and wait for explicit user approval.

## Decisions And Assumptions

## Progress Log
```

## Execute One Stage At A Time

1. Mark only the approved current stage as `In Progress`.
2. Implement only that stage's scope.
3. Run the planned verification and any newly necessary focused checks.
4. Update the plan with actual deliverables, commands, results, deviations, and unresolved issues.
5. Mark the stage `Awaiting Review`, not `Completed`.
6. Update `.codex/memory-bank/active_context.md` with the current state and next action.
7. Update the smallest relevant durable memory:
   - `decisions.md` for accepted architectural, scientific, or evaluation choices;
   - `experiments.md` for reproducible experiment settings, metrics, and observations;
   - `glossary.md` for durable project terminology.
8. Give the user a compact review packet containing:
   - what changed;
   - files or artifacts;
   - verification evidence;
   - known limitations or deviations;
   - the exact approval needed.
9. Stop. Do not start the next stage in the same turn.

After explicit approval, mark the reviewed stage `Completed`, append the approval to the progress
log, set the next stage to `In Progress`, update memory, and execute only that next stage.

If the user rejects or requests changes, keep the stage `Awaiting Review` or return it to
`In Progress`, implement the revisions, re-run verification, and present it again. Never infer
approval from silence or from approval of a different artifact.

## Visualize Results With Jupyter

For stages that produce quantitative, scientific, diagnostic, benchmark, or model results that
benefit from visualization:

1. Invoke and follow `data-analytics:jupyter-notebooks`.
2. Run its mandatory Data Analytics user-context preflight before notebook work.
3. Create or update a reproducible notebook under `notebooks/`.
4. Keep parameters, source paths, seeds, preprocessing, splits, and assumptions visible.
5. Include focused tables and labeled charts plus reasonableness checks.
6. Execute the notebook top-to-bottom before presenting the stage for review.
7. Record the notebook path and execution status in the stage review packet and plan progress log.

Do not create a notebook for a purely structural change with no meaningful result to inspect.
Never use an unexecuted notebook output as evidence for a conclusion.

## Status Rules

Use only these stage states:

- `Pending`: approved but not started.
- `In Progress`: currently being implemented.
- `Awaiting Review`: implemented and verified, waiting for the user.
- `Completed`: explicitly approved by the user.
- `Blocked`: cannot proceed; record the blocker and required resolution.

Use only these plan states:

- `draft`: not yet approved.
- `approved`: approved and not yet started.
- `in_progress`: at least one stage is active or awaiting review.
- `completed`: every stage is explicitly approved.
- `blocked`: progress cannot continue.

When the final stage is approved, mark the plan `completed`, set `Next stage: complete`, update
`active_context.md`, and provide a concise final summary with verification and artifact paths.

## Guardrails

- Preserve leakage boundaries and scientific validity in every stage.
- Do not rewrite past progress silently. Append dated corrections or explain status changes.
- Do not claim verification that was not run.
- Do not expand a stage's scope without updating the plan and obtaining approval when the expansion
  changes deliverables, risk, or acceptance criteria.
- Keep raw data unchanged and place derived artifacts in generated locations.
