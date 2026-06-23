# Thesis Defense Presentation

Status: in_progress
Last updated: 2026-06-23
Next stage: complete Stage 5 after review

## Goal

Create a thesis defense presentation in Russian that clearly and honestly presents the
master thesis on reconstructing binary visual stimuli from multichannel EEG with machine
learning. The presentation should foreground the thesis's strongest contribution: a
strict, reproducible, leakage-aware experimental pipeline and a conservative
interpretation of near-chance reconstruction results.

## Scope

- A new presentation source, expected at `latex/presentation.tex`.
- A built presentation PDF, expected at `latex/output/presentation.pdf`.
- Reuse of existing thesis figures and experiment plots from `latex/images/`.
- Optional concise speaker notes or rehearsal bullets if needed for defense delivery.

## Exclusions

- No new model training or data preprocessing runs.
- No changes to the thesis text unless a separately approved correction is needed.
- No wholesale redrawing of scientific figures when existing thesis assets are suitable.
- No overstated interpretation of near-chance reconstruction metrics.

## Acceptance Criteria

- The slide deck covers the topic, relevance, goal and tasks, dataset, 6x6 reconstruction
  formulation, feature/model families, evaluation protocols, main results, BNCI external
  pipeline checks, limitations, and final conclusion.
- Numerical claims match the thesis, reviewer text, or persisted experiment evidence.
- The presentation explicitly explains leakage control and why strict evaluation leads to
  a conservative result.
- The final PDF builds successfully.
- Slides are readable, not overloaded, and suitable for an approximately 7-10 minute
  defense talk.

## Stages

### 1. Narrative And Slide Outline - Completed

- Objective: Define the defense story and ordered slide list before writing the deck.
- Deliverables:
  - a 10-12 slide outline with each slide's title, purpose, and key talking point;
  - a short narrative arc for the defense.
- Constraints:
  - present strict leakage-aware evaluation as a positive methodological contribution;
  - keep the near-chance main-task results honest;
  - avoid framing BNCI experiments as external validation of the reconstruction task.
- Verification:
  - check the outline against `latex/chapters/introduction.tex`,
    `latex/chapters/experiments/base.tex`, `latex/chapters/conclusion.tex`,
    `review_supervisor.txt`, and `review_external_reviewer.txt`;
  - confirm every planned quantitative claim has a source.
- Completion criteria: The outline is coherent enough that rejecting or changing one
  slide does not invalidate the whole plan.
- Review gate: Stop and wait for explicit user approval.
- Actual deliverables:
  - created `../notes/thesis-defense-presentation-outline.md`;
  - drafted a 12-slide defense outline with a narrative arc, key thesis facts,
    assumptions, open questions, slide-by-slide purpose, key talking points, visual
    candidates, numeric claims, and source references.
- Completed verification:
  - checked the outline against `latex/chapters/introduction.tex`,
    `latex/chapters/methodology/base.tex`, `latex/chapters/experiments/base.tex`,
    `latex/chapters/conclusion.tex`, `review_supervisor.txt`, and
    `review_external_reviewer.txt`;
  - `rg` confirmed key quantitative claims are present in the thesis/review sources:
    180 main observations, `0.518382`, `0.512011`, `0.273000`, `0.385224`,
    `0.764062`, `0.711260`, and `0.850690`;
  - `rg` found no `Data_Pattern`, `labels.json`, `schema-v`, `TODO`, `FIXME`, or
    `PLACEHOLDER` markers in the outline;
  - a static whitespace/conflict-marker check passed for the outline and memory-bank
    files;
  - `git -C code diff --check` passed for the code memory-bank files.

### 2. Presentation Skeleton And Visual Inventory - Completed

- Objective: Create the technical presentation scaffold and choose the visual assets.
- Deliverables:
  - `latex/presentation.tex`;
  - a visual inventory inside the plan progress log or review packet;
  - an initial built PDF with placeholder slide bodies if needed.
- Constraints:
  - use an academic, defense-appropriate visual style;
  - prefer existing thesis images from `latex/images/`;
  - keep slide text short and readable.
- Verification:
  - run a LaTeX build for the presentation;
  - check that every referenced image exists;
  - inspect the generated PDF for obvious layout failures.
- Completion criteria: The presentation source builds into a readable PDF skeleton with
  the agreed slide order.
- Review gate: Stop and wait for explicit user approval.
- Actual deliverables:
  - created `../latex/presentation.tex`, a 12-slide Beamer presentation skeleton
    following the approved Stage 1 outline;
  - created `../notes/thesis-defense-visual-inventory.md`, listing primary and backup
    visual assets;
  - built `../latex/output/presentation.pdf` with 12 pages.
- Completed verification:
  - confirmed all `\includegraphics` targets in `latex/presentation.tex` exist under
    `latex/images/`;
  - built the deck with
    `docker run --rm -i -v /home/slauva/Projects/master-thesis-2024-2026/latex:/work -w /work ghcr.io/xu-cheng/texlive-full latexmk -synctex=1 -interaction=nonstopmode -file-line-error -pdf -outdir=./output presentation`;
  - `pdfinfo latex/output/presentation.pdf` reported 12 pages, 16:9 page size, and a
    nonzero file size;
  - `pdftotext` confirmed extractable Russian slide text;
  - log search found no LaTeX errors, missing image files, undefined references,
    overfull/underfull boxes, fatal errors, or emergency stops;
  - visually inspected rendered PNG previews for slides 1, 6, 9, and 10;
  - `git -C latex diff --check -- presentation.tex` passed;
  - static whitespace/conflict-marker checks passed for the presentation source, visual
    inventory, and memory-bank files.
- Known limitations:
  - this is a skeleton deck, not the final content-polished presentation;
  - the build log contains a non-fatal T2A sans-serif bold font substitution warning.

### 3. Main Slide Content - Completed

- Objective: Fill the deck with the core thesis content up through methodology and
  experimental setup.
- Deliverables:
  - completed slides for relevance, goal/tasks, dataset, formulation, preprocessing,
    features, models, and validation protocols.
- Constraints:
  - keep wording in Russian;
  - preserve terminology from the thesis where possible;
  - explain the 36-pixel multi-output formulation without turning the slide into a
    chapter excerpt.
- Verification:
  - build the PDF;
  - search for stale implementation jargon inappropriate for defense slides;
  - compare terms and claims with methodology and experiment chapters.
- Completion criteria: The deck tells the problem and method story clearly before the
  result section.
- Review gate: Stop and wait for explicit user approval.
- Actual deliverables:
  - revised `../latex/presentation.tex` slides 2-8 with fuller core thesis content:
    relevance and scientific question, goal/object/subject, reconstruction formulation,
    dataset and analysis unit, leakage-aware evaluation contour, input representations,
    model families, and evaluation protocols/metrics.
  - rebuilt `../latex/output/presentation.pdf` with the updated 12-slide deck.
- Completed verification:
  - compared the new slide claims against `latex/chapters/introduction.tex`,
    `latex/chapters/methodology/base.tex`, and `latex/chapters/experiments/base.tex`;
  - verified source support for `180` main observations, `[0.5, 15.5)` seconds, `125`
    Hz, `141/39` cross-subject split sizes, `81/81` bidirectional cross-trial sizes,
    FFT/Welch, Morlet, Superlet, STFT, EEGNet, DeepConvNet, ShallowConvNet, bootstrap,
    and Holm correction;
  - built the deck with Docker/latexmk and produced `../latex/output/presentation.pdf`;
  - `pdfinfo` reported 12 pages and a nonzero file size;
  - `pdftotext` confirmed extractable Russian text;
  - log search found no LaTeX errors, missing image files, undefined references,
    overfull/underfull boxes, fatal errors, or emergency stops;
  - searched `../latex/presentation.tex` for stale internal terms and placeholders:
    no `Data_Pattern`, `labels.json`, `schema-v`, `upstream`, `outputs/`,
    `experiments/`, `TODO`, `FIXME`, `PLACEHOLDER`, or user-facing `pipeline` /
    `train-` wording remained, except `experiment_pipeline.pdf` as an image filename;
  - visually inspected rendered PNG previews for slides 5, 6, 7, and 8;
  - `git -C latex diff --check -- presentation.tex` passed;
  - static whitespace/conflict-marker checks passed for the presentation source and
    memory-bank files.
- Known limitations:
  - Stage 3 intentionally focused on methodology/setup slides; result, limitation, and
    conclusion framing are still scheduled for Stage 4;
  - the build log still contains a non-fatal T2A sans-serif bold font substitution
    warning.

### 4. Results, Limitations, And Defense Framing - Completed

- Objective: Add the result, limitation, and conclusion slides with careful scientific
  framing.
- Deliverables:
  - result slides for cross-subject and bidirectional cross-trial protocols;
  - a BNCI pipeline-check slide;
  - limitation and conclusion slides.
- Constraints:
  - do not claim statistically reliable superiority over Logistic Regression;
  - present `balanced accuracy` values near 0.5 as conservative evidence under strict
    evaluation;
  - present BNCI2014-001 and BNCI2014-009 as independent EEG pipeline checks, not as a
    replacement for external validation on the same reconstruction task.
- Verification:
  - build the PDF;
  - verify the main numbers against `latex/chapters/experiments/base.tex` and
    `latex/chapters/conclusion.tex`;
  - visually inspect plots and tables for readability.
- Completion criteria: The presentation has an honest, defensible final scientific
  message.
- Review gate: Stop and wait for explicit user approval.
- Actual deliverables:
  - revised `../latex/presentation.tex` slides 9-12 with careful result framing:
    near-chance reconstruction results, reconstruction examples, BNCI2014-001/009
    contour checks, limitations, conclusion, and future-work framing;
  - rebuilt `../latex/output/presentation.pdf` with the updated 12-slide deck.
- Completed verification:
  - checked Stage 4 numerical claims against `latex/chapters/experiments/base.tex`
    and `latex/chapters/conclusion.tex`: `0.518382`, `[0.486733; 0.561786]`,
    `0.512011`, `[0.500668; 0.520872]`, `0.273000`, `0.385224`, `0.764062`,
    `0.711260`, and `0.850690`;
  - built the deck with Docker/latexmk and produced `../latex/output/presentation.pdf`;
  - `pdfinfo` reported 12 pages, 16:9 page size, and a nonzero file size;
  - `pdftotext` confirmed the new result, BNCI, and conclusion slide text is
    extractable;
  - log search found no LaTeX errors, missing image files, undefined references,
    overfull/underfull boxes, fatal errors, or emergency stops;
  - visually inspected rendered PNG previews for slides 9, 10, 11, and 12;
  - searched `../latex/presentation.tex` for stale internal terms and placeholders:
    no `Data_Pattern`, `labels.json`, `schema-v`, `upstream`, `outputs/`,
    `experiments/`, `TODO`, `FIXME`, `PLACEHOLDER`, or user-facing `pipeline` /
    `пайплайн` wording remained, except `experiment_pipeline.pdf` as an image
    filename;
  - `git -C latex diff --check -- presentation.tex` passed;
  - static whitespace/conflict-marker checks passed for the presentation source and
    memory-bank files.
- Known limitations:
  - Stage 5 still needs final whole-deck QA and optional speaking bullets;
  - the build log still contains a non-fatal T2A sans-serif bold font substitution
    warning.

### 5. Build, Layout QA, And Rehearsal Packet - Awaiting Review

- Objective: Produce the final defense-ready artifact and compact speaking support.
- Deliverables:
  - final `latex/output/presentation.pdf`;
  - optional speaking bullets or a short talk script;
  - updated plan progress log and active context.
- Constraints:
  - do not silently change scientific claims during layout cleanup;
  - keep final slides within the intended defense time;
  - record any skipped checks explicitly.
- Verification:
  - run the final LaTeX build;
  - check the build log for LaTeX errors, missing references, missing figures, and fatal
    warnings;
  - inspect the final PDF for text overflow, missing images, and overloaded slides.
- Completion criteria: The deck is ready for a defense rehearsal and all stage evidence is
  recorded.
- Review gate: Stop and wait for explicit user approval.
- Actual deliverables:
  - final rebuilt deck at `../latex/output/presentation.pdf`;
  - compact speaking/rehearsal bullets at `../notes/thesis-defense-speaking-bullets.md`;
  - updated staged plan and active context.
- Completed verification:
  - ran the final Docker/latexmk build; the deck was already up to date and
    `../latex/output/presentation.pdf` remained the final artifact;
  - `pdfinfo` reported 12 pages, 16:9 page size, and a nonzero file size;
  - `pdftotext` confirmed all 12 slide titles are extractable from the PDF;
  - log search found no LaTeX errors, missing image files, undefined references,
    overfull/underfull boxes, fatal errors, or emergency stops;
  - confirmed all `\includegraphics` assets referenced by `../latex/presentation.tex`
    exist under `../latex/images/`;
  - rendered all 12 slides to PNG previews and visually inspected slides 1-12 for
    missing images, text overlap, and obvious overflow;
  - searched the presentation and speaking bullets for stale internal terms and
    placeholders: no `Data_Pattern`, `labels.json`, `schema-v`, `upstream`, `outputs/`,
    `experiments/`, `TODO`, `FIXME`, `PLACEHOLDER`, or user-facing `pipeline` /
    `пайплайн` wording remained, except `experiment_pipeline.pdf` as an image filename;
  - `git -C latex diff --check -- presentation.tex` passed;
  - `git -C code diff --check` passed for the staged-plan and active-context files;
  - static whitespace/conflict-marker checks passed for the presentation source,
    speaking bullets, and memory-bank files.
- Known limitations:
  - the build log still contains a non-fatal T2A sans-serif bold font substitution
    warning;
  - ImageMagick `montage` was unavailable, so whole-deck visual QA used individually
    rendered PNG previews instead of a contact sheet.
- Review revision:
  - after user feedback to rework slide 12, replaced the final slide with a stronger
    defense-oriented conclusion: "what was done", "what the data showed", "why it
    matters", "future work", and a final takeaway;
  - updated `../notes/thesis-defense-speaking-bullets.md` to match the revised final
    slide;
  - rebuilt `../latex/output/presentation.pdf` and visually inspected the new slide 12;
  - log search found no LaTeX errors, missing image files, undefined references,
    overfull/underfull boxes, fatal errors, or emergency stops after the revision.
- 20-minute script revision:
  - created `../notes/thesis-defense-20min-script.md`, a full Russian defense script
    aligned to the then-current 12-slide deck and the conservative result framing;
  - the script includes a timing table, slide-by-slide speech text, and a compact final
    phrase;
  - `wc -w` reported 2448 words, suitable for approximately 18-20 minutes with slide
    transitions and pauses;
  - verified the script contains the key numerical claims from the deck and thesis:
    `180`, `141/39`, `81/81`, `0.518382`, `0.512011`, `0.273000`, `0.385224`,
    `0.764062`, `0.711260`, and `0.850690`;
  - searched the script for stale internal terms and placeholders: no `Data_Pattern`,
    `labels.json`, `schema-v`, `upstream`, `outputs/`, `experiments/`, `TODO`,
    `FIXME`, `PLACEHOLDER`, or user-facing `pipeline` / `пайплайн` wording;
  - static whitespace/conflict-marker check passed for the script.
- Content and language revision:
  - after user feedback, proofread the Russian wording in `../latex/presentation.tex`
    with attention to gender, number, case, and English/Russian terminology;
  - expanded the introduction from one relevance slide into a broader context sequence:
    EEG task difficulty by type, why reconstruction from imagery is difficult, and the
    specific scientific question;
  - added a problem statement slide with literature examples (`EEG-to-output` review,
    DreamDiffusion, Dijkstra et al./Xie et al.) showing why methodological strictness is
    important for EEG-to-image work;
  - split the methodology into signal representations and model/training settings, adding
    concrete values for Logistic Regression, classical baselines, EEGNet/DeepConvNet/
    ShallowConvNet, `BCEWithLogitsLoss`, AdamW, learning rate `10^{-3}`, weight decay
    `10^{-4}`, batch size `16`, maximum `300` epochs, early stopping `30`, gradient clip
    `1.0`, and seeds `42/43/44`;
  - reused `eeg_rhythms_frequency_bands.png` from the thesis images in the expanded
    introductory context;
  - rebuilt `../latex/output/presentation.pdf`; `pdfinfo` reported 15 pages, 16:9 page
    size, and a nonzero file size;
  - log search found no LaTeX errors, missing image files, undefined references,
    overfull/underfull boxes, fatal errors, or emergency stops after the revision;
  - rendered the revised PDF to PNG previews and visually inspected the new/changed
    slides for context, problematics, dataset wording, and methodology settings;
  - `git -C latex diff --check -- presentation.tex` passed;
  - static whitespace/conflict-marker checks passed for `../latex/presentation.tex`;
  - stale internal-term search passed except for `experiment_pipeline.pdf` as an image
    filename.
  - updated `../notes/thesis-defense-20min-script.md` to match the expanded 15-slide
    deck; `wc -w` reported 2321 words, and checks found no stale internal terms,
    placeholders, trailing whitespace, or conflict markers.

## Decisions And Assumptions

- The presentation language is Russian.
- The deck is now expanded to 15 slides after user-requested context/methodology
  revision; a separate 20-minute script exists for the longer defense format.
- The likely technical format is LaTeX/Beamer under `latex/`, reusing the existing thesis
  LaTeX environment and image assets.
- The main defense narrative should be: the task is hard, the pipeline is strict and
  reproducible, the current dataset does not support strong reconstruction claims, and
  independent BNCI checks show the software pipeline can detect signal on established EEG
  tasks.

## Progress Log

- 2026-06-23: Plan approved by the user and persisted as the active staged plan.
- 2026-06-23: Stage 1 implemented and verified. The outline was saved to
  `../notes/thesis-defense-presentation-outline.md` and Stage 1 is awaiting user
  review.
- 2026-06-23: User approved Stage 1 by requesting "дальше"; Stage 1 marked completed
  and Stage 2 started.
- 2026-06-23: Stage 2 implemented and verified. The Beamer skeleton was saved to
  `../latex/presentation.tex`, the visual inventory to
  `../notes/thesis-defense-visual-inventory.md`, and the PDF skeleton to
  `../latex/output/presentation.pdf`. Stage 2 is awaiting user review.
- 2026-06-23: User approved Stage 2 by requesting "дальше"; Stage 2 marked completed
  and Stage 3 started.
- 2026-06-23: Stage 3 implemented and verified. Slides 2-8 were revised in
  `../latex/presentation.tex`, the deck was rebuilt to
  `../latex/output/presentation.pdf`, and Stage 3 is awaiting user review.
- 2026-06-23: User approved Stage 3 by requesting "дальше"; Stage 3 marked completed
  and Stage 4 started.
- 2026-06-23: Stage 4 implemented and verified. Slides 9-12 were revised in
  `../latex/presentation.tex`, the deck was rebuilt to
  `../latex/output/presentation.pdf`, and Stage 4 is awaiting user review.
- 2026-06-23: User approved Stage 4 by requesting "дальше"; Stage 4 marked completed
  and Stage 5 started.
- 2026-06-23: Stage 5 implemented and verified. The final PDF remained at
  `../latex/output/presentation.pdf`, rehearsal bullets were saved to
  `../notes/thesis-defense-speaking-bullets.md`, all slides were visually inspected,
  and Stage 5 is awaiting user review.
- 2026-06-23: User requested a revision of slide 12. The final slide was reworked,
  `../latex/output/presentation.pdf` was rebuilt, speaking bullets were updated, and
  Stage 5 is awaiting user review again.
- 2026-06-23: User requested a 20-minute defense text. Created
  `../notes/thesis-defense-20min-script.md`, verified length and key claims, and Stage 5
  is awaiting user review again.
- 2026-06-23: User requested stronger Russian proofreading, a broader introduction,
  clearer problematics, and more detailed methodology. Revised
  `../latex/presentation.tex`, expanded the deck to 15 slides, rebuilt
  `../latex/output/presentation.pdf`, verified clean LaTeX log/diff checks, visually
  inspected revised slides, and Stage 5 is awaiting user review again.
