# Logistic Regression Evaluation Protocols And CLI

Status: in_progress
Last updated: 2026-06-15
Next stage: 2 - Evaluation Protocols And Runner

## Goal

Extend the random-imagery pixel-wise Logistic Regression baseline with IoU and Hamming loss,
scientifically distinct cross-subject and within-subject evaluation protocols, a reusable
experiment runner, a terminal CLI, and an executable training notebook.

## Scope

- Preserve the existing `Data_Pattern/patt`, `type="random"`, full `[0.5, 15.5)` epoch, and
  36-pixel target contracts.
- Keep the current fixed 26/7 subject split as the primary cross-subject protocol.
- Add an identity-overlapping bidirectional cross-trial protocol for the 27 subjects that have
  both trials.
- Add schema-v2 evaluation artifacts while retaining read compatibility with the existing
  schema-v1 run `f515948b6bf5af55`.
- Provide equivalent `uv run logistic-regression ...` and
  `uv run python -m experiments.logistic_regression ...` entry points.
- Keep `notebooks/5.0-logistic-regression-random-pixels.ipynb` as the reference artifact report
  and add a separate training notebook.

## Acceptance Criteria

- IoU and Hamming loss are tested, reported for the model and every baseline, and satisfy their
  mathematical invariants.
- Cross-subject evaluation preserves disjoint subjects and the established 141/39 row split.
- Within-subject evaluation uses both Trial 1 -> Trial 2 and Trial 2 -> Trial 1 directions with
  train-only screening and tuning in each direction.
- No direction overlaps train/test sample keys, trials, random seeds, or image payloads.
- CLI can train, reuse an immutable run, and evaluate one or more persisted runs without
  implicitly loading joblib.
- The training notebook executes both protocols through reusable code rather than owning training
  logic.
- Ruff, the full test suite, notebook integration checks, and `git diff --check` pass.

## Stages

### 1. Reconstruction Metrics - Completed

- Objective: Add IoU and normalized Hamming loss to the reusable evaluation contract and reference
  report.
- Deliverables:
  - extend `PredictionMetrics` with per-sample IoU, mean sample IoU, global micro IoU, and Hamming
    loss;
  - update metric validation, exports, baseline tables, and the executed `5.0` notebook;
  - add synthetic metric tests and current-run regression checks.
- Constraints:
  - IoU treats binary value `1` as foreground;
  - an empty target and empty prediction have sample IoU `1.0`;
  - IoU and Hamming loss do not participate in feature selection or hyperparameter tuning;
  - preserve balanced accuracy as the primary selection metric.
- Verification:
  - perfect, partial-overlap, disjoint, and empty-empty IoU cases;
  - `hamming_loss == 1 - bit_accuracy`;
  - `hamming_loss == mean_hamming_distance / 36`;
  - reference run values: mean sample IoU `0.335257970`, micro IoU `0.334634146`, and Hamming
    loss `0.485754986`;
  - execute `5.0` top-to-bottom and run focused tests, Ruff, and diff check.
- Completion criteria: all model and baseline summaries include both IoU aggregations and Hamming
  loss with deterministic validated values.
- Review gate: Stop and wait for explicit user approval.

### 2. Evaluation Protocols And Runner - In Progress

- Objective: Define and execute cross-subject and within-subject protocols through one reusable
  orchestration layer.
- Deliverables:
  - typed protocol and direction schemas;
  - protocol-specific split builders and leakage audits;
  - one runner for dataset, targets, feature screening, grid search, prediction, baselines,
    bootstrap, and evaluation;
  - synthetic ordering and leakage tests.
- Cross-subject protocol:
  - retain `GroupShuffleSplit(test_size=0.2, random_state=42)`;
  - retain 141 train rows from 26 subjects and 39 test rows from 7 disjoint subjects;
  - bootstrap complete held-out subject clusters.
- Within-subject protocol:
  - include only subjects with both trials: 27 subjects and 162 total rows;
  - exclude subjects `14, 24, 27, 28, 29, 32` with provenance;
  - run Trial 1 train -> Trial 2 test and Trial 2 train -> Trial 1 test independently;
  - each direction has 81 train and 81 test rows from the same 27 subject identities;
  - run separate train-only feature screening and all 36 grid searches for each direction;
  - keep five-fold inner CV grouped by subject;
  - combine both directions only after their independent predictions are complete;
  - bootstrap all six out-of-trial predictions belonging to each sampled subject.
- Constraints:
  - call the secondary protocol `identity-overlapping bidirectional cross-trial`;
  - do not describe it as a separate subject-specific model;
  - outer-test features remain inaccessible until all train-only decisions for that direction are
    complete;
  - no overlap in trial rows, sample keys, seeds, or complete image payloads.
- Verification:
  - real-corpus counts, eligibility, class support, and leakage audits;
  - both directions contain both classes for all 36 tasks;
  - test-access ordering tests;
  - deterministic synthetic runner tests.
- Completion criteria: the runner returns complete protocol results with separate directions and
  a valid combined within-subject summary.
- Review gate: Stop and wait for explicit user approval.

### 3. Evaluation Artifacts And CLI - Pending

- Objective: Persist protocol-aware evaluations and expose safe terminal commands.
- Deliverables:
  - schema-v2 artifact payload with `evaluation.json`;
  - backward-compatible schema-v1 metadata/array reader;
  - `argparse` CLI and project script entry point;
  - CLI tests for parsing, execution, reuse, output, and failures.
- Artifact contract:
  - store protocol, direction, eligible and excluded subjects, split audit, model metrics,
    baseline metrics, and bootstrap summary;
  - keep exact SHA-256 inventory and manifest-last atomic publication;
  - store each within-subject direction as its own immutable run;
  - combine complementary directions during evaluation without rewriting either run.
- CLI contract:
  - `uv run logistic-regression run --protocol cross-subject`;
  - `uv run logistic-regression run --protocol within-subject`;
  - `uv run logistic-regression evaluate RUN_DIR [RUN_DIR ...] [--json]`;
  - equivalent `python -m experiments.logistic_regression` commands;
  - support `--config PATH`, repeatable `--set KEY=VALUE`, and `--reuse-existing`;
  - refuse an existing config hash by default;
  - `--reuse-existing` validates and reuses the run without fitting;
  - do not provide destructive overwrite behavior;
  - `evaluate` validates metadata and arrays without loading joblib.
- Constraints:
  - no new CLI dependency;
  - human output includes run path, protocol, split, selected feature family, model metrics, and
    baselines;
  - `--json` emits deterministic machine-readable output.
- Verification:
  - schema-v1 reference-run evaluation;
  - schema-v2 round trip and corruption rejection;
  - duplicate refusal and valid reuse;
  - dotted OmegaConf override parsing;
  - multi-run within-subject aggregation;
  - CLI exit codes and JSON output.
- Completion criteria: a user can train or evaluate both protocols entirely from the terminal
  without editing Python source.
- Review gate: Stop and wait for explicit user approval.

### 4. Training Notebook And Final Comparison - Pending

- Objective: Provide an executable notebook entry point for training and a clear comparison of
  the two evaluation protocols.
- Deliverables:
  - executed `notebooks/5.1-logistic-regression-training.ipynb`;
  - notebook integration test;
  - final protocol comparison table and uncertainty figure;
  - updated experiment memory and final repository verification.
- Notebook contract:
  - expose config path, dotted overrides, protocol selection, and `REUSE_EXISTING=True` near the
    top;
  - default to cross-subject plus both within-subject directions;
  - call the reusable runner and never duplicate fit, split, screening, persistence, or metric
    logic;
  - show split/leakage audits, excluded within-subject participants, screening results, grid-search
    summaries, artifact provenance, metrics, and baselines;
  - show direction-level and combined within-subject results separately;
  - compare cross-subject and within-subject in one table and one uncertainty chart;
  - state that the protocol difference is not a pure model effect because the generalization
    targets and evaluation populations differ.
- Constraints:
  - `5.0` remains the reference report for run `f515948b6bf5af55`;
  - repeated execution reuses valid immutable runs when configured;
  - unexecuted notebook output is not accepted as evidence.
- Verification:
  - execute `5.1` top-to-bottom;
  - visual inspection of all figures;
  - integration marker, execution counts, and no error outputs;
  - `uv run ruff check .`;
  - `uv run pytest`;
  - `git diff --check`.
- Completion criteria: both protocols can be launched from the notebook and their persisted,
  source-backed results are compared without overstating within-subject performance.
- Review gate: Stop and wait for explicit user approval.

## Decisions And Assumptions

- Use standard-library `argparse`; add no CLI framework dependency.
- Cross-subject fixed 80/20 remains the primary subject-generalization estimate.
- Within-subject means identity-overlapping cross-trial transfer, not per-subject model fitting.
- Subjects with only one trial are scientifically ineligible for the bidirectional cross-trial
  protocol and are excluded rather than block-split.
- Feature-family screening and hyperparameter search are repeated independently inside each outer
  training partition.
- Report protocols separately and additionally provide a shared comparison table and chart.
- Do not average cross-subject and within-subject results into one overall score.
- Raw FIF and label files remain unchanged; all new outputs are generated artifacts.

## Progress Log

- 2026-06-15: User requested IoU, Hamming loss, terminal CLI, and a notebook that can run training.
- 2026-06-15: User selected CLI `run + evaluate`, a separate `5.1` training notebook, both sample
  and micro IoU, both CLI entry forms, immutable duplicate refusal with explicit reuse, and full
  notebook execution with reuse.
- 2026-06-15: User added cross-subject and within-subject evaluation requirements.
- 2026-06-15: Real-corpus audit confirmed 27 two-trial subjects with 81 rows per trial direction,
  complete class support, and no cross-trial seed or image overlap.
- 2026-06-15: User selected fixed 80/20 cross-subject evaluation, bidirectional cross-trial
  within-subject evaluation, and separate results plus a shared comparison.
- 2026-06-15: User explicitly requested that the approved plan be saved.
- 2026-06-15: User requested execution; Stage 1 marked In Progress.
- 2026-06-15: Stage 1 implemented. `PredictionMetrics` now includes per-sample IoU, mean sample
  IoU, micro IoU, and normalized Hamming loss with aggregate consistency validation.
- 2026-06-15: Added synthetic perfect/partial/disjoint/empty-empty IoU coverage and immutable-run
  regression checks. Reference model values are mean sample IoU `0.335257970`, micro IoU
  `0.334634146`, and Hamming loss `0.485754986`.
- 2026-06-15: Re-executed `notebooks/5.0-logistic-regression-random-pixels.ipynb`; all 9 code
  cells ran without errors, the verification marker is present, and all 5 figures were visually
  inspected.
- 2026-06-15: Verification passed: `uv run ruff check .`, `uv run pytest` (`259 passed`, two
  pre-existing multiprocessing warnings), and `git diff --check`. Stage 1 is Awaiting Review;
  Stage 2 has not started.
- 2026-06-15: User approved Stage 1 with `next`; Stage 1 marked Completed and Stage 2 marked
  In Progress.
