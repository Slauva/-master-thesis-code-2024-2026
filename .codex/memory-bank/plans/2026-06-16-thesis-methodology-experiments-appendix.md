# Thesis Methodology, Experiments, And Appendix Writing

Status: completed
Last updated: 2026-06-16
Current stage: 8 - Static QA And Memory Update (completed)
Next stage after review: optional LaTeX compilation after separate user permission

## Goal

Align the thesis text with the completed source-backed experiment work in `code/`: update the
methodology, write the experiments chapter, and add appendix material for feature groups and deep
learning architectures without overstating the results.

## Scope

- Update `latex/chapters/methodology/base.tex`.
- Rewrite `latex/chapters/experiments/base.tex`.
- Extend `latex/chapters/appendix/base.tex`.
- Add BibTeX entries only when the source metadata is checked and the citation is needed.
- Use existing notebooks, memory-bank entries, and JSON/NumPy artifacts; do not run new experiments
  as part of this writing plan.

## Acceptance Criteria

- Thesis chapters contain no placeholders such as `\ldots`, `TODO`, or `f_s = ...`.
- `Data_Train` is not described as a machine-learning train split.
- The methodology describes the ready visual-imagery/random-stimulus subset in thesis-facing terms,
  without exposing internal names such as `Data_Pattern`, `patt`, `.fif`, `labels.json`, or `img`.
- The methodology does not present BrainBERT, GAN, or diffusion models as completed experiments.
- The methodology states that the used data were already preprocessed and summarizes the upstream
  preprocessing pipeline from `visual_stimuli_reconstruction_thesis.pdf` or its checked note.
- Numerical claims are traceable to experiment artifacts or executed notebooks.
- FFT/Welch, Morlet, Superlet, STFT, LNDP, 1D-LGP, 1D-LBP, EEGNet, DeepConvNet, and
  ShallowConvNet have checked citation anchors before they are cited in thesis text.
- Deep learning architectures, spectral transforms, and local-pattern feature groups are described
  clearly enough to support appendix diagrams, implementation references, and short examples.
- Every new `\cite{...}` key exists in `latex/biblio/bibliography.bib`.

## Stages

### 1. Evidence Inventory - Completed

- Objective: collect the source-backed facts needed for the methodology, experiments chapter, and
  appendix.
- Deliverables:
  - `code/.codex/memory-bank/thesis-writing-evidence.md`.
- Constraints:
  - do not edit LaTeX chapters;
  - do not run new experiments;
  - keep facts separate from assumptions and writing decisions.
- Verification:
  - inspected `code/.codex/memory-bank/active_context.md`, `decisions.md`, and `experiments.md`;
  - inspected selected `results.json`, `evaluation.json`, and notebook markers;
  - checked existing bibliography/source-map coverage for feature and architecture citations;
  - checked `git status --short` in `code/`.
- Completion criteria: the evidence inventory lists the stable facts, artifact paths, metrics,
  citation anchors, and open citation gaps needed by later stages.
- Review gate: approved by the user's 2026-06-16 request to continue the plan.

### 2. Methodology Update - Completed

- Objective: bring chapter 2 in line with the actual pipeline.
- Deliverables:
  - updated `latex/chapters/methodology/base.tex`.
- Constraints:
  - describe only completed or actually used model families as experiment methods;
  - keep BrainBERT and generative models as background or future work only;
  - preserve leakage boundaries and the random-imagery task definition while hiding internal
    dataset/storage names from thesis-facing prose;
  - describe the upstream preprocessed-data pipeline from the checked predecessor-thesis note;
  - cite spectral transforms, local-pattern features, and CNN architectures only with verified
    BibTeX keys.
- Verification:
  - compared against `code/.codex/memory-bank/thesis-writing-evidence.md`,
    `active_context.md`, `decisions.md`, `experiments.md`,
    `code/confs/features/default.yaml`, `code/experiments/random_imagery/registry.py`,
    and `code/experiments/random_imagery_torch/models.py`;
  - static check found no `\ldots`, `TODO`, `f_s`, stale corpus counts, or stale
    preprocessing claims in `latex/chapters/methodology/base.tex`;
  - static check confirmed BrainBERT/GAN/diffusion are not presented as completed
    experimental methods;
  - existing methodology citation keys were verified in `latex/biblio/bibliography.bib`;
  - 2026-06-16 revision removed internal `Data_Pattern/patt`, `.fif`, `labels.json`, and `img`
    wording from `latex/chapters/methodology/base.tex`;
  - 2026-06-16 revision added checked BibTeX entries for predecessor thesis, FFT/Welch, Morlet,
    Superlet, STFT/Gabor, EEGNet, and DeepConvNet/ShallowConvNet citations.
- Completion criteria: chapter 2 matches the code/artifact reality and no longer promises
  unexecuted experiments.
- Review gate: approved by the user's 2026-06-16 request to continue the plan.

### 3. Feature Appendix Structure - Completed

- Objective: describe all feature groups and source anchors.
- Deliverables:
  - appendix section covering time, spectral, spatial, and local-pattern feature groups.
- Constraints:
  - use Jaiswal and Banka for LNDP, 1D-LGP, and 1D-LBP;
  - cite the spectral transforms used in code: FFT/Welch, Morlet, Superlet, and STFT/Gabor;
  - use covariance/Riemannian EEG sources for covariance, correlation, and log-covariance;
  - use checked preprocessing/spectral sources already present in the thesis source map where
    possible;
  - include implementation anchors to the relevant project modules and a short reproducible example
    or pseudocode sketch for each nontrivial transform family.
- Verification:
  - compare against `code/features/*.py` and `code/confs/features/default.yaml`;
  - compare spectral-transform descriptions against `code/preprocessors/*.py` and executed
    notebooks `2.1`--`2.5`;
  - verify all citation keys in `latex/biblio/bibliography.bib`.
- Completed verification:
  - compared the appendix text against `code/features/classical.py`,
    `code/features/local_patterns.py`, `code/features/config.py`,
    `code/features/windowing.py`, `code/confs/features/default.yaml`,
    `code/preprocessors/fft.py`, `morlet.py`, `superlet.py`, `stft.py`, and
    `code/confs/preprocessing/*.yaml`;
  - added implementation anchors and pseudocode examples for feature extraction, FFT/Welch,
    Morlet, Superlet, STFT, LNDP, 1D-LGP, and 1D-LBP;
  - verified newly used appendix citation keys exist in `latex/biblio/bibliography.bib`;
  - static check found no `TODO`, `\ldots`, or `f_s =` placeholders in the methodology and
    appendix files.
- Completion criteria: the appendix describes every feature family in enough detail for thesis
  reproducibility, code traceability, examples, and future diagrams.
- Review gate: approved by the user's 2026-06-16 request to continue the plan.

### 4. Deep Learning Architecture Appendix - Completed

- Objective: describe the deep learning model structures and diagram prompts/specifications.
- Deliverables:
  - appendix section for EEGNet, DeepConvNet, ShallowConvNet, EEGNet-SSVEP, and EEGNet-v1;
  - diagram prompts/specs with Russian labels.
- Constraints:
  - distinguish primary full-experiment architectures from exploratory ports;
  - describe these PyTorch modules as spectral-input adaptations, not numerically identical
    TensorFlow translations;
  - cite Lawhern et al. for EEGNet and Schirrmeister et al. for DeepConvNet/ShallowConvNet.
- Verification:
  - compare against `code/experiments/random_imagery_torch/models.py`;
  - compare citation/attribution text against `ARL_EEGMODELS_NOTICE.md`;
  - verify citation keys or record missing BibTeX gaps.
- Completed verification:
  - compared the appendix text against `code/experiments/random_imagery_torch/models.py`,
    `code/experiments/random_imagery_torch/config.py`,
    `code/experiments/random_imagery_torch/ARL_EEGMODELS_NOTICE.md`,
    `code/.codex/memory-bank/thesis-writing-evidence.md`, and
    `code/.codex/memory-bank/decisions.md`;
  - added a deep-learning appendix section for EEGNet, DeepConvNet, ShallowConvNet,
    EEGNet-SSVEP, and EEGNet-v1;
  - described all five modules as PyTorch spectral-input adaptations with 36-logit heads,
    not numerically identical TensorFlow/Keras translations;
  - distinguished EEGNet, DeepConvNet, and ShallowConvNet as the primary full-comparison
    architectures from EEGNet-SSVEP and EEGNet-v1 as exploratory ports;
  - added Russian-labelled diagram specifications for the architecture figures;
  - verified all newly used architecture citation keys exist in `latex/biblio/bibliography.bib`;
  - static check found no `TODO`, `\ldots`, or `f_s =` placeholders in the methodology and
    appendix files.
- Completion criteria: the appendix is sufficient to generate architecture figures and cite the
  relevant papers correctly.
- Review gate: approved by the user's 2026-06-16 request to continue the plan.

### 5. Experiments Chapter Draft - Completed

- Objective: rewrite chapter 3 without placeholders.
- Deliverables:
  - updated `latex/chapters/experiments/base.tex`.
- Constraints:
  - do not claim statistically significant superiority of any learned model over Logistic
    Regression;
  - keep cross-subject and cross-trial/within-subject protocols separate.
- Verification:
  - verify split sizes, protocols, metrics, bootstrap intervals, and final comparison values
    against artifacts and executed notebooks.
- Completed verification:
  - compared the chapter text against `code/.codex/memory-bank/thesis-writing-evidence.md`,
    `code/.codex/memory-bank/experiments.md`, `code/.codex/memory-bank/decisions.md`,
    `code/experiments/random_imagery/config.py`, `code/experiments/random_imagery/registry.py`,
    and selected persisted `evaluation.json` artifacts;
  - verified cross-subject split sizes, within-subject direction sizes, excluded cross-trial
    subjects, Logistic Regression reference metrics, descriptive Ridge and Torch leaders,
    bootstrap intervals, exact-match values, and final Holm-adjusted p-value;
  - rewrote `latex/chapters/experiments/base.tex` without placeholders and without claiming
    statistically reliable superiority over Logistic Regression;
  - kept cross-subject and bidirectional cross-trial protocols separate;
  - used `tabularx`/`L{...}`/`Y` for new thesis tables;
  - static check found no `TODO`, `\ldots`, `f_s =`, old metric placeholders, internal dataset
    storage names, or old fixed-width `tabular` environments in the edited thesis chapter set.
- Completion criteria: chapter 3 contains data, implementation, protocols, results, and limitations.
- Review gate: approved by the user's 2026-06-16 request to continue the plan.

### 6. Tables, Figures, And Prompts - Completed

- Objective: add thesis-ready tables and figure placeholders/specs.
- Deliverables:
  - protocol, feature, model, and result tables;
  - captions, labels, and prompts/specs for missing figures.
- Constraints:
  - use only real `predictions.npy` and `test_targets.npy` for reconstruction examples;
  - do not invent visual examples or unverified values.
- Verification:
  - check LaTeX labels;
  - check numerical values against the evidence inventory and artifacts.
- Completed verification:
  - added source-backed input representation, model-family, bidirectional cross-trial, and
    reconstruction-example tables to `latex/chapters/experiments/base.tex`;
  - reconstructed the example rows only from real
    `code/artifacts/experiments/random-imagery-torch/shallow-convnet-morlet-multilabel/678f75c694c69eb2/arrays/test_targets.npy`
    and `predictions.npy` arrays;
  - checked DeepConvNet STFT direction metrics against persisted `evaluation.json` artifacts and
    combined metrics against executed notebooks `6.0` and `6.1`;
  - verified new labels `tab:experiment-input-representations`,
    `tab:experiment-model-families`, `tab:within-subject-results`,
    and `tab:reconstruction-examples-real`;
  - removed repository paths and the working-only future-figure specification table from the
    thesis text after user review; figure-generation ideas should stay in planning notes until
    actual figures are created;
  - static check found no `TODO`, `\ldots`, `f_s =`, old metric placeholders, internal dataset
    storage names, repository paths, backtick-formatted paths, or fixed-width `tabular`
    environments in the edited thesis chapter set.
- Completion criteria: every table and figure has a source path or a precise generation spec.
- Review gate: approved by the user's 2026-06-16 request to continue the plan.

### 7. Programmatic Thesis Figures - Completed

- Objective: generate real thesis figures and insert them into LaTeX without exposing repository
  paths in the thesis prose.
- Deliverables:
  - result ranking figure for balanced accuracy across the final comparison;
  - paired-delta figure versus Logistic Regression with uncertainty and Holm-adjusted context;
  - reconstruction examples figure using real target/prediction arrays;
  - experiment pipeline figure;
  - architecture figures for EEGNet, DeepConvNet, and ShallowConvNet, plus optional exploratory
    EEGNet-SSVEP/EEGNet-v1 figures only if they are actually referenced in the appendix;
  - LaTeX `figure` environments with thesis-facing captions and labels.
- Constraints:
  - use `matplotlib` for all numeric result figures and reconstruction grids;
  - use deterministic programmatic drawing for pipeline and architecture schemes, preferably
    `matplotlib` patches or Graphviz if already available, not AI-generated raster images;
  - reuse the visual style of the executed project notebooks: clean white background, restrained
    gridlines, consistent sans-serif labels, readable Russian captions/axis titles, muted colors,
    no decorative gradients, and export-ready sizing;
  - save figures as vector PDF where possible, and PNG only when raster output is materially clearer;
  - build figures from saved experiment artifacts and executed notebooks only; do not run new
    experiments;
  - keep repository paths, array filenames, notebook names, and implementation module paths out of
    thesis prose and captions;
  - preserve leakage-aware interpretation: separate cross-subject and bidirectional cross-trial
    protocols, and do not imply statistically reliable superiority over Logistic Regression.
- Verification:
  - check every plotted value against persisted `evaluation.json` artifacts, final comparison tables,
    or executed notebooks;
  - check that reconstruction examples come from real target/prediction arrays;
  - inspect generated images for readable text, consistent styling, and no clipped labels;
  - verify every inserted `\includegraphics` target exists and every new label is unique;
  - run static search to confirm no repository paths were added to thesis text.
- Completed verification:
  - added deterministic Matplotlib figure generator
    `code/scripts/generate_thesis_figures.py`;
  - generated nine PDF figures under `latex/images/`: cross-subject and bidirectional
    cross-trial balanced-accuracy rankings, paired deltas versus Logistic Regression,
    real reconstruction examples, the experiment pipeline, and EEGNet/DeepConvNet/
    ShallowConvNet architecture schemes;
  - figure generation reloaded saved artifacts only, used real `test_targets.npy` and
    `predictions.npy` for reconstruction examples, and verified the final comparison
    anchors `ridge-regression-independent:0.518382`,
    `deep-convnet-stft-multilabel:0.512011`, and `min_holm_p=0.273000`;
  - inserted thesis-facing `figure` environments into
    `latex/chapters/experiments/base.tex` and `latex/chapters/appendix/base.tex`
    without repository paths in captions or prose;
  - replaced the appendix table of future architecture-diagram specifications with
    real architecture figures;
  - visually inspected PNG previews for the cross-subject ranking, reconstruction
    examples, pipeline, and EEGNet architecture figure;
  - verified all `\includegraphics` targets exist, all new figure labels are unique,
    and `uv run ruff check scripts/generate_thesis_figures.py` passes;
  - static search found no `TODO`, `\ldots`, `f_s =`, internal dataset storage names,
    repository artifact paths, or notebook paths in the edited thesis chapter set.
- Completion criteria: the thesis contains actual generated figures, not future-figure specs, and
  all figure sources are tracked in the plan or generation script rather than in thesis prose.
- Review gate: approved by the user's 2026-06-16 request to continue the plan.

### 8. Static QA And Memory Update - Completed

- Objective: perform final static checks and update durable memory.
- Deliverables:
  - static QA report;
  - updated plan progress log and active context.
- Constraints:
  - run LaTeX compilation only after separate user permission.
- Verification:
  - `rg` checks for placeholders and stale wording;
  - citation-key check against `latex/biblio/bibliography.bib`;
  - final evidence traceability check for major numerical claims.
- Completed verification:
  - static search found no `TODO`, `\ldots`, `f_s =`, internal dataset storage names,
    repository artifact paths, notebook paths, or backtick-formatted code/path fragments in
    `latex/chapters/methodology/base.tex`, `latex/chapters/experiments/base.tex`, and
    `latex/chapters/appendix/base.tex`;
  - static search found no fixed-width `\begin{tabular}` environments in the edited thesis
    chapter set;
  - verified 22 LaTeX labels with no duplicates, 9 `\includegraphics` targets with no missing
    files, and 15 citation keys with no missing BibTeX entries;
  - `uv run ruff check scripts/generate_thesis_figures.py` passes;
  - checked major numerical claims against `code/.codex/memory-bank/thesis-writing-evidence.md`,
    `code/.codex/memory-bank/experiments.md`, and the assert-backed figure-generation script,
    including 180 random-imagery rows, 141/39 cross-subject split, 81/81 cross-trial directions,
    162 held-out rows from 27 identities, cross-subject leader `0.518382`, combined cross-trial
    leader `0.512011`, and final minimum Holm-adjusted p-value `0.273000`;
  - did not run LaTeX compilation because the plan requires separate user permission.
- Completion criteria: chapters are ready for user review and later LaTeX build.
- Review gate: Stop and wait for explicit user approval.

## Decisions And Assumptions

- No new experiments will be run in this plan.
- The internal main experimental task is random-imagery reconstruction over `Data_Pattern/patt`,
  but thesis-facing methodology prose should call this the ready visual-imagery/random-stimulus
  subset rather than exposing internal dataset/storage names.
- The main scientific conclusion is conservative: results are near chance, and descriptive leaders
  are not reliable superiority claims.
- The primary full Torch comparison architectures are EEGNet, DeepConvNet, and ShallowConvNet.
- EEGNet-SSVEP and EEGNet-v1 may be described as implemented/tested exploratory ports, not as
  primary full-experiment models.
- Future thesis tables should use `tabularx` with total width `\textwidth`, the reusable `L{...}`
  and `Y` column types from `latex/settings/preamble.tex`, reduced `\tabcolsep`, and ragged-right
  wrapping instead of fixed-width `tabular` layouts that can overflow page margins.
- Thesis figures should be generated programmatically from artifacts with the same general visual
  style as the project notebooks. AI image generation is not appropriate for numeric/scientific
  figures in this plan.
- User review is required after every stage before the next stage begins.

## Progress Log

- 2026-06-16: User requested a staged plan for methodology, experiments, feature appendix, and
  deep-learning architecture appendix.
- 2026-06-16: User added the requirement to describe deep learning structures, diagram prompts, and
  all feature groups with article links.
- 2026-06-16: User requested persistence and stage tracking through `manage-staged-plans`.
- 2026-06-16: User explicitly approved implementation of the plan.
- 2026-06-16: Saved the plan, registered it in active context, executed Stage 1, and created
  `code/.codex/memory-bank/thesis-writing-evidence.md`.
- 2026-06-16: User requested continuing the saved plan; Stage 1 was treated as approved.
- 2026-06-16: Executed Stage 2 and rewrote `latex/chapters/methodology/base.tex` to match the
  verified random-imagery pipeline, dataset counts, feature/model families, leakage boundaries,
  and conservative interpretation limits. Stage 2 entered review.
- 2026-06-16: User requested Stage 2 revisions: remove internal dataset/storage names from
  methodology, describe the predecessor-thesis preprocessing pipeline, add checked citations for
  spectral transforms, local-pattern features, and CNN architectures, and expand later appendix
  stages to include implementation anchors and examples. Reopened/revised Stage 2 accordingly.
- 2026-06-16: User requested continuing the saved plan; Stage 2 was treated as approved. Executed
  Stage 3 and added the feature appendix section to `latex/chapters/appendix/base.tex`, covering
  time, spectral, spatial, and local-pattern feature groups, implementation anchors, formulas,
  citation anchors, and pseudocode examples for the implemented transform families. Stage 3
  entered review.
- 2026-06-16: User requested continuing the saved plan; Stage 3 was treated as approved. Executed
  Stage 4 and added the deep-learning architecture appendix section to
  `latex/chapters/appendix/base.tex`, covering EEGNet, DeepConvNet, ShallowConvNet, EEGNet-SSVEP,
  EEGNet-v1, spectral-input adaptation notes, primary/exploratory status, and Russian-labelled
  diagram specifications. Stage 4 entered review.
- 2026-06-16: User reported overflowing appendix tables. Converted all thesis chapter tables to
  `tabularx` with `\textwidth`, added reusable `L{...}` and `Y` column types, and recorded this
  layout rule for future tables.
- 2026-06-16: User requested the next plan stage; Stage 4 was treated as approved. Executed Stage 5
  and rewrote `latex/chapters/experiments/base.tex` around the verified random-imagery
  experiment subset, implementation, protocols, metrics, conservative result interpretation,
  cross-subject and bidirectional cross-trial summaries, and limitations. Stage 5 entered review.
- 2026-06-16: User requested continuing the saved plan; Stage 5 was treated as approved. Executed
  Stage 6 and added thesis-ready input, model, within-subject result, real reconstruction example,
  and future-figure specification tables to `latex/chapters/experiments/base.tex`. Stage 6
  entered review.
- 2026-06-16: User reviewed Stage 6 and rejected repository paths and the future-figure
  specification table in thesis prose. Removed code/artifact paths from the thesis chapters,
  removed the future-figure specification table from chapter 3, and kept only thesis-facing
  tables.
- 2026-06-16: User asked whether thesis figures will be generated by the agent or through
  matplotlib. Recorded Stage 7 as a dedicated programmatic figure-generation stage using
  matplotlib/deterministic drawing in the style of the existing project notebooks, with no
  AI-generated scientific figures.
- 2026-06-16: User requested continuing the saved plan; Stage 6 was treated as approved. Executed
  Stage 7 by adding a deterministic Matplotlib figure-generation script, generating thesis-ready
  PDF figures from saved artifacts, inserting them into chapter 3 and appendix A, replacing future
  architecture figure specifications with real figures, and verifying figure targets, labels,
  visual previews, artifact-backed numerical anchors, and static no-path/no-placeholder checks.
  Stage 7 entered review.
- 2026-06-16: User requested continuing the saved plan; Stage 7 was treated as approved. Executed
  Stage 8 static QA: cleaned one false-positive LaTeX quote/backtick fragment in the appendix,
  rechecked placeholders, stale internal wording, fixed-width tables, figure targets, labels,
  citation keys, Ruff status for the figure generator, and evidence traceability for major
  numerical claims. LaTeX compilation was not run because it requires separate permission. Plan
  completed and active context updated for final review / optional later build.
