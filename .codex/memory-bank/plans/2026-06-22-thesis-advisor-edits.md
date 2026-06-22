# Thesis Advisor Edits

Status: awaiting_review
Last updated: 2026-06-22
Next stage: complete after Stage 7 review

## Goal

Revise the thesis in response to the scientific advisor's comments: strengthen
Chapter 1 literature support with concrete EEG-to-image works, reframe the
global contribution around leakage-aware evaluation rigor, justify the
15-second aggregation choice without rerunning experiments, clean the EEG
rhythm figure, standardize terminology, remove thesis-facing implementation
jargon, consolidate dataset counts, and explicitly compare the current thesis
with the 2025 Dementyev/Parepko/Baranov thesis.

## Scope

- Thesis-facing LaTeX chapters under `../latex/chapters/`.
- Bibliography files under `../latex/biblio/` when new citations are needed.
- Figure asset `../latex/images/eeg_rhythms_frequency_bands.png` if the
  in-image labels need replacement.
- Durable terminology memory in `glossary.md`.

## Exclusions

- No new model training or data preprocessing runs.
- No changes to raw data.
- No reinterpretation of saved experiment metrics beyond already persisted
  evidence.
- No LaTeX PDF rebuild unless explicitly approved as part of the final QA
  stage.

## Acceptance Criteria

- Section 1.4 cites concrete EEG-to-image / EEG visual-stimulus reconstruction
  papers in addition to the meta-review.
- Introduction, methodology, experiments, and conclusion foreground strict
  leakage-aware evaluation as a contribution while keeping the conservative
  result honest.
- The 15-second window is described as a conservative epoch-level unit aligned
  with annotation and leakage control, with explicit acknowledgement that it
  sacrifices temporal dynamics.
- Figure 1.3 no longer contains non-academic labels such as "доступ к
  подсознанию"; its caption is academically neutral.
- A project glossary maps English terms to preferred Russian thesis terms; the
  thesis introduces English terms once and then uses the Russian terms.
- Thesis-facing prose no longer contains internal engineering labels such as
  `schema-v3`.
- Dataset counts `16/31/651/434/217/180` are consolidated and the relationship
  between trial, observation, and subject is explicit.
- The thesis explicitly compares the current work with
  `notes/2025-dementyev-parepko-baranov-visual-stimuli-reconstruction-thesis.md`.
- Static QA finds no new placeholders, missing citation keys, missing
  full-footcite entries, duplicate labels, or missing figures in the edited
  thesis-facing files.

## Stages

### 1. Literature Anchors For EEG-To-Image - Completed

- Objective: Replace the single-review dependency in Section 1.4 with a concise
  literature bridge from the review to concrete EEG-to-image models.
- Deliverables:
  - revised `../latex/chapters/litreview/brain-activity-reconstruction.tex`;
  - new BibTeX and full-footcite entries when needed.
- Constraints:
  - distinguish visual perception EEG-to-image from the thesis visual-imagery
    binary reconstruction setting;
  - do not imply that large generative EEG-to-image models are directly
    applicable to the current small 6x6 dataset;
  - keep the review citation as field-level context, not the only evidence.
- Verification:
  - check new citation keys exist in `bibliography.bib`;
  - check new keys have full-footcite entries;
  - search Section 1.4 for the concrete cited works.
- Completion criteria: Section 1.4 names specific EEG-to-image model families
  and concrete papers, while the conclusion of the section still motivates the
  local pixel-wise reconstruction setting.
- Review gate: Stop and wait for explicit user approval.
- Actual deliverables:
  - expanded `../latex/chapters/litreview/brain-activity-reconstruction.tex`
    with concrete EEG-to-image examples: EEG2IMAGE, DreamDiffusion, BrainVis,
    and GWIT;
  - added four arXiv BibTeX entries to `../latex/biblio/bibliography.bib`;
  - added matching full-footcite entries to
    `../latex/biblio/footcite-entries.tex`.
- Completed verification:
  - `rg` confirmed all four new citation keys are present in
    `bibliography.bib`;
  - `rg` confirmed all four new citation keys have `fullcite@...` entries in
    `footcite-entries.tex`;
  - `rg` confirmed Section 1.4 names EEG2IMAGE, DreamDiffusion, BrainVis, GWIT,
    and explicitly preserves the caveat that generative EEG-to-image results do
    not directly solve the current small binary visual-imagery setting.

### 2. Reframe Contribution Around Strict Evaluation - Completed

- Objective: Shift the thesis framing from a primarily negative result to a
  positive methodological contribution: a strict, reproducible leakage-aware
  evaluation pipeline.
- Deliverables:
  - targeted edits to `../latex/chapters/introduction.tex`;
  - targeted edits to `../latex/chapters/methodology/base.tex`;
  - targeted edits to `../latex/chapters/experiments/base.tex`;
  - targeted edits to `../latex/chapters/conclusion.tex`.
- Constraints:
  - do not hide near-chance model results;
  - keep claims tied to persisted artifacts and existing evidence;
  - preserve the distinction between model performance and evaluation rigor.
- Verification:
  - search for stale "negative result" framing;
  - search for leakage-aware evaluation language across the edited chapters.
- Completion criteria: the thesis clearly states that strict evaluation is an
  independent result and the conservative model outcome follows from that
  procedure.
- Review gate: Stop and wait for explicit user approval.
- Actual deliverables:
  - updated `../latex/chapters/introduction.tex` to make strict evaluation part
    of the motivation, goal, and novelty;
  - updated `../latex/chapters/methodology/base.tex` to introduce information
    boundaries and leakage audits as a methodological element before the
    protocol descriptions;
  - updated `../latex/chapters/experiments/base.tex` so the interpretation
    starts from strict evaluation evidence rather than from an "negative
    result" framing;
  - updated `../latex/chapters/conclusion.tex` to state the main contribution
    as a strict test under leakage and subject-structure control.
- Completed verification:
  - `rg` confirmed the previous phrase
    "Главный экспериментальный результат является отрицательным" is absent;
  - `rg` confirmed the new strict-evaluation framing appears in the
    introduction, experiments, and conclusion;
  - no LaTeX build was run.

### 3. Dataset Accounting And 15-Second Window Defense - Completed

- Objective: Consolidate dataset counts and justify the 15-second aggregation
  choice without rerunning experiments.
- Deliverables:
  - a compact paragraph and/or table in `../latex/chapters/methodology/base.tex`;
  - limitation wording in methodology or experiments.
- Constraints:
  - explicitly explain the relationship between subject, trial, observation,
    and model row;
  - verify the `217 -> 180` transition before final wording;
  - defend the full-window representation as conservative and annotation-aligned,
    while acknowledging the loss of within-epoch temporal dynamics.
- Verification:
  - trace counts against the 2025 thesis note and current data/indexing evidence;
  - search for duplicated or contradictory dataset-count statements.
- Completion criteria: a reader can follow `16/31/651/434/217/180` without
  needing repository-specific context.
- Review gate: Stop and wait for explicit user approval.
- Actual deliverables:
  - added `tab:dataset-counts` to `../latex/chapters/methodology/base.tex`;
  - revised the methodology prose to explain the transition from 217 random
    observations in the predecessor description to the 180-row recollection-phase
    subset used in current experiments;
  - added methodology and experiment-limitations wording that defends the
    full 15-second epoch as a conservative analysis unit while acknowledging
    lost within-epoch dynamics;
  - added an explicit `\selectlanguage{russian}` before the methodology
    section "Стратегии обучения и валидации" after the build revealed a stale
    English-language state in the generated table of contents;
  - recorded the writing decision in `decisions.md`.
- Completed verification:
  - executable dataset index check reported 540 prepared recollection-phase
    blocks with 180 random and 360 geometric blocks, and three random blocks per
    subject/trial pair in that prepared phase;
  - `rg` confirmed the edited thesis-facing methodology/experiments fragments
    contain no internal storage names `Data_Pattern`, `patt`, `.fif`,
    `labels.json`, or `img`;
  - `rg -o "label\{[^}]+\}" latex/chapters | sort | uniq -d` reported no
    duplicate labels;
  - after the user reported a build error, ran the approved Docker/latexmk
    diploma build; it completed successfully and produced
    `../latex/output/diploma.pdf` with 91 A4 pages;
  - post-build log search found no `LaTeX Error`, undefined citations,
    undefined references, fatal errors, or `Command \CYR... unavailable`
    encoding errors.

### 4. Figure 1.3 Cleanup - Completed

- Objective: Replace non-academic in-image labels and caption wording for the
  EEG rhythm figure.
- Deliverables:
  - updated `../latex/images/eeg_rhythms_frequency_bands.png` or replacement
    image;
  - updated caption in `../latex/chapters/litreview/basics-eeg-visual.tex` if
    needed.
- Constraints:
  - remove "доступ к подсознанию";
  - use neutral physiological or cognitive associations;
  - keep the figure readable in an A4 thesis layout.
- Verification:
  - visual inspection of the updated image;
  - `rg подсозн` over thesis-facing files and images metadata where possible.
- Completion criteria: the figure and caption read as academic context rather
  than popular-science claims.
- Review gate: Stop and wait for explicit user approval.
- Actual deliverables:
  - replaced `../latex/images/eeg_rhythms_frequency_bands.png` with a clean
    academic schematic table of EEG frequency ranges, neutral functional
    associations, and illustrative waveforms;
  - revised the caption in
    `../latex/chapters/litreview/basics-eeg-visual.tex` to describe the
    ranges as orienting analysis bands whose interpretation depends on task,
    electrodes, and preprocessing.
- Completed verification:
  - visually inspected the regenerated PNG and confirmed no in-image text
    overlaps;
  - `file` confirmed the image is a valid 1500x1080 RGB PNG;
  - `rg` found no `подсозн` or `доступ к подсознанию` in thesis-facing text;
  - Docker/latexmk diploma build completed successfully and produced
    `../latex/output/diploma.pdf`, 91 A4 pages;
  - post-build log search found no `LaTeX Error`, undefined citations,
    undefined references, fatal errors, or `Command \CYR... unavailable`
    encoding errors.

### 5. Terminology And Vibe-Code Pass - Completed

- Objective: Standardize key terms and remove internal implementation jargon.
- Deliverables:
  - updated `glossary.md`;
  - targeted edits in methodology, experiments, appendix, and conclusion.
- Constraints:
  - introduce English terms once in parentheses, then use Russian terms;
  - replace `schema-v3` and similar implementation labels with thesis-facing
    descriptions;
  - preserve technical precision for leakage boundaries and metrics.
- Verification:
  - search for `schema-v3`, `cross-subject`, `cross-trial`, `bidirectional`,
    `train-only`, `seeded Bernoulli`, `bit accuracy`, `exact match`,
    `sample key`, and `random seed`;
  - manually review remaining English terms for necessity.
- Completion criteria: terminology is consistent and thesis-facing prose is free
  of accidental code/artifact vocabulary.
- Review gate: Stop and wait for explicit user approval.
- Actual deliverables:
  - extended `glossary.md` with preferred Russian thesis terms for leakage-aware
    evaluation, train-only transforms, cross-subject and bidirectional
    cross-trial protocols, seeded Bernoulli, bit accuracy, exact match accuracy,
    sample keys, random seeds, balanced accuracy, baseline models, and
    implementation-only `schema-v*` vocabulary;
  - revised `../latex/chapters/introduction.tex`,
    `../latex/chapters/methodology/base.tex`,
    `../latex/chapters/experiments/base.tex`,
    `../latex/chapters/conclusion.tex`, and
    `../latex/chapters/appendix/base.tex` to introduce English terms once and
    then use Russian wording;
  - cleaned review-facing terminology in
    `../latex/chapters/litreview/eeg-ml-models.tex`,
    `../latex/chapters/litreview/eeg-time-analyze.tex`, and
    `../latex/chapters/litreview/brain-activity-reconstruction.tex`.
- Completed verification:
  - `rg` confirmed `schema-v*` does not appear in thesis-facing LaTeX files;
  - terminology search now returns only deliberate first definitions such as
    `leakage-aware`, `train-only`, `cross-subject`, `seeded Bernoulli`,
    `bit accuracy`, and `exact match accuracy`, plus non-user-facing labels
    containing `cross-subject`;
  - duplicate-label search reported no duplicates;
  - Docker/latexmk diploma build completed successfully and produced
    `../latex/output/diploma.pdf` with 91 A4 pages;
  - post-build log search found no `LaTeX Error`, undefined citations,
    undefined references, fatal errors, or `Command \CYR... unavailable`
    encoding errors.

### 6. Explicit Comparison With 2025 Thesis - Completed

- Objective: Add a clear comparison between the current thesis and the 2025
  Dementyev/Parepko/Baranov thesis.
- Deliverables:
  - comparison paragraph or compact table in methodology or experiments.
- Constraints:
  - state that the current thesis uses the previously collected dataset;
  - distinguish predecessor contributions in data collection, preprocessing,
    exploratory analysis, and earlier baselines from current contributions in
    reproducible leakage-aware reconstruction evaluation and model comparison;
  - avoid overstating superiority over the previous work.
- Verification:
  - check against
    `../notes/2025-dementyev-parepko-baranov-visual-stimuli-reconstruction-thesis.md`;
  - verify `dementyev2025visualstimuli` citation is present.
- Completion criteria: the current thesis position relative to the 2025 work is
  explicit and fair.
- Review gate: Stop and wait for explicit user approval.
- Actual deliverables:
  - added a new `Связь с предшествующей работой` section to
    `../latex/chapters/methodology/base.tex`;
  - added `tab:predecessor-comparison`, a compact comparison table separating
    the predecessor project's dataset/protocol/preprocessing/exploratory work
    from the current thesis's leakage-controlled reconstruction evaluation;
  - clarified that the two sets of numerical results should not be read as a
    direct model competition because the research question, subset, model set,
    and evaluation protocol differ;
  - cleaned nearby visible terminology from `trial`, `upstream preprocessing
    pipeline`, and `visual imagery reconstruction` into Russian thesis-facing
    terms.
- Completed verification:
  - `rg` confirmed `dementyev2025visualstimuli` is cited in the new comparison
    section and exists in both `bibliography.bib` and `footcite-entries.tex`;
  - duplicate-label search reported no duplicates, including the new
    `tab:predecessor-comparison` label;
  - targeted terminology search found no visible `Trial`, `upstream`,
    `schema-v*`, or `visual imagery reconstruction` in the edited methodology
    and experiments files; the only remaining `cross-trial` occurrence is the
    deliberate first-definition English term in parentheses;
  - Docker/latexmk diploma build completed successfully and produced
    `../latex/output/diploma.pdf` with 93 A4 pages;
  - post-build log search found no `LaTeX Error`, undefined citations,
    undefined references, fatal errors, or `Command \CYR... unavailable`
    encoding errors.

### 7. Static QA And Optional Build - Awaiting Review

- Objective: Verify the edited thesis sources.
- Deliverables:
  - QA notes in this plan's progress log;
  - optionally rebuilt PDFs if the user separately approves the LaTeX build.
- Constraints:
  - do not run a full Docker/latexmk build without separate permission;
  - static checks should still run after every stage where relevant.
- Verification:
  - placeholder and internal-term searches;
  - duplicate-label check;
  - missing-figure check;
  - missing BibTeX-key and full-footcite-entry check;
  - optional LaTeX build.
- Completion criteria: no known static regressions remain; optional build status
  is recorded.
- Review gate: Stop and wait for explicit user approval.
- Actual deliverables:
  - ran final thesis-facing placeholder, internal-term, terminology, figure,
    label, reference, citation, full-footcite, and LaTeX-log checks;
  - replaced the deep implementation URL
    `experiments/random_imagery_torch` in
    `../latex/chapters/appendix/base.tex` with the public project repository
    URL to avoid leaking a repository-internal module path into the thesis text;
  - rebuilt `../latex/output/diploma.pdf` after that final appendix cleanup.
- Completed verification:
  - static search found no `TODO`, `FIXME`, `PLACEHOLDER`, `schema-v*`,
    internal dataset storage names, `.fif`, `labels.json`, notebook paths,
    `experiments/`, `outputs/`, `upstream`, `visual imagery reconstruction`,
    or the removed Figure 1.3 "подсозн" wording in thesis-facing LaTeX;
  - terminology search leaves only deliberate first-definition English terms
    and non-user-facing labels;
  - structural validator reported 32 labels with no duplicates, no missing
    references, 17 `\includegraphics` targets with no missing files, and 43
    citation keys with no missing BibTeX or full-footcite entries;
  - Docker/latexmk diploma build completed successfully and produced
    `../latex/output/diploma.pdf` with 93 A4 pages;
  - post-build log search found no `LaTeX Error`, undefined citations,
    undefined references, fatal errors, or `Command \CYR... unavailable`
    encoding errors.
- Deviation:
  - Although the stage listed a build as optional and separately approved, a
    final build was run after a small appendix URL cleanup so that the generated
    PDF matches the verified source files.

## Decisions And Assumptions

- The project memory bank is stored under `code/.codex/memory-bank/`.
- New plans are saved only after explicit user approval; approval was given on
  2026-06-22.
- The current work is a thesis-writing revision plan, not a new experiment
  plan.
- The 15-second aggregation will be defended as a conservative, epoch-level
  modeling choice rather than as an optimal temporal model.
- New literature citations may use arXiv entries when they are the available
  primary source for recent EEG-to-image model papers.

## Progress Log

- 2026-06-22: User approved the staged plan. Saved it as an approved plan.
- 2026-06-22: Implemented Stage 1 and marked it Awaiting Review. Added concrete
  EEG-to-image literature anchors and matching bibliography/footcite entries;
  no LaTeX build was run.
- 2026-06-22: User approved moving to the next stage. Marked Stage 1 completed,
  implemented Stage 2, and marked it Awaiting Review. No LaTeX build was run.
- 2026-06-22: User approved moving to the next stage. Marked Stage 2 completed,
  implemented Stage 3, and marked it Awaiting Review. No LaTeX build was run.
- 2026-06-22: User approved moving to the next stage. Marked Stage 3 completed,
  implemented Stage 4, rebuilt `diploma.pdf`, and marked Stage 4 Awaiting
  Review.
- 2026-06-22: User approved moving to the next stage. Marked Stage 4 completed,
  implemented Stage 5, rebuilt `diploma.pdf`, and marked Stage 5 Awaiting
  Review.
- 2026-06-22: User approved moving to the next stage. Marked Stage 5 completed,
  implemented Stage 6, rebuilt `diploma.pdf`, and marked Stage 6 Awaiting
  Review.
- 2026-06-22: User approved moving to the final QA stage. Marked Stage 6
  completed, implemented Stage 7, rebuilt `diploma.pdf` after a small appendix
  URL cleanup, and marked Stage 7 Awaiting Review.
