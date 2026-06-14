# Active Context

## Current Focus

Initial project agent setup:

- Create a project-level agent guide.
- Add a memory bank that can be updated during future work.
- Add local skills for EEG/ML research, project memory, and Python optimization/review.
- Preserve the contents of `python_optimization_prompt.md` inside a skill reference before that source file is removed.

## Next Actions

- Build or review a dataset index for `data/`.
- Define canonical train/validation/test split policy.
- Decide how labels from `labels.json` map to targets.
- Add a minimal test strategy once dataset utilities exist.

## Open Questions

- What is the exact prediction target?
- Should evaluation be leave-one-subject-out, grouped K-fold by subject, within-subject, or multiple protocols?
- Which recordings are considered training, pattern/reference, validation, or test data?
