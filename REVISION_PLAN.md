# Revision Plan

## Objective

Turn the current manuscript into the strongest defensible version of the paper:

1. keep the finite-memory cube-root law as the rigorous theoretical backbone;
2. make `CI`, `CI^E`, and coercive masking the flagship contribution;
3. tighten citation chains, proof language, and calibration framing so the paper reads like a finished acceptance-ready revision rather than a technically strong but slightly dispersed draft.

This plan is driven by the current manuscript state, the artifact bundle already in the repository, and a targeted web verification pass on the most relevant missing citations.

## Current State

### What is already strong

- The cube-root story is already technically well defended in the manuscript.
- The lower-bound proposition for window-restricted estimators is already materially stronger than a sliding-window-only result.
- The manuscript already contains healthy-only calibration, paired bootstrap intervals, and a consolidated sensitivity table in the body, not only the appendix.
- The artifact bundle already supports the paper's honest regime split:
  - passive streams: generic changepoint detectors remain the stronger default alarms;
  - active/logged settings: `CI^E` is the distinctive contribution.

### What the evidence says right now

- `ELEC2`: `ADWIN` has the highest matched-lead count in `experiments/artifacts/elec2/elec2_summary.csv`.
- `Bikes`: `ADWIN` again leads the passive benchmark by matched leads in `experiments/artifacts/bikes/bikes_summary.csv`.
- `KuaiRand`: `CI^E` improves bubble detection from `0.458` to `0.692` and collapse detection from `0.548` to `0.700`, with paired bootstrap deltas excluding zero.
- `Particle`: at influence `0.3` and effort penalty `lambda = 3.0`, `CI` remains high while `CI^E` drops materially, creating a masking gap `0.252`.
- `Sinkhorn runtime`: the current Gaussian artifact already makes the practical `epsilon` trade-off visible and supports the manuscript's operational framing.

The repository-level audit generated from current artifacts is saved at:

- `experiments/artifacts/revision/revision_audit.md`

## Web-Verified Citation Chain

### Verified and already used in the manuscript

- `Perdomo et al. (ICML 2020)`: performative stability / performative prediction.
- `Hardt and Mendler-D"unner (2023)`: performative prediction overview and framing.
- `Goldfeld et al.`: now verified as published in `Electronic Journal of Statistics, 18(1), 2024`, DOI `10.1214/24-EJS2217`.

### Verified and added to `bibliography.bib` for likely use

- `Farina and Perdomo (2026)`: stability of online algorithms in performative prediction.
- `Stiennon et al. (NeurIPS 2020)`: TL;DR / RLHF summarization benchmark.
- `Bai et al. (2022)`: HH-RLHF assistant training with policy-KL tracking.
- `Dubois et al. (2024 arXiv / NeurIPS 2023 spotlight)`: AlpacaFarm as a practical feedback-learning simulation benchmark.

## Flagship Positioning

### Final paper stance

The strongest version of the paper is:

- the cube-root law explains the finite-memory floor under drift;
- `CI^E` is the paper's flagship contribution for action-coupled and logged regimes;
- passive benchmarks are retained precisely to show honesty of regime separation, not to claim universal dominance.

### Editorial principle

The paper should read as:

- `theory contribution`: finite-memory tracking floor;
- `diagnostic contribution`: intervention-aware masking detection;
- `evidence split`: passive streams favor generic drift alarms, while active/logged regimes are where the proposed diagnostics become necessary.

## File-by-File Edit Plan

### 1. `frontmatter/abstract.tex`

Goal: make the two-part contribution explicit, with `CI^E` as the distinctive practical contribution.

Edits:

- keep the cube-root law in the first half;
- explicitly state that passive detectors remain stronger on passive streams;
- foreground the active/logged result as the main practical differentiation.

Target effect:

- the abstract should answer, in one read, why the paper matters beyond another drift-detection paper.

### 2. `frontmatter/introduction.tex`

Goal: make the paper's scope and flagship contribution impossible to misread.

Edits:

- keep the current scope caveat, but move the intervention-aware contribution into a more prominent role;
- state explicitly that the paper does not seek universal superiority on passive streams;
- keep the cube-root law as structural context for memory tuning.

### 3. `discussion/related_work.tex`

Goal: tighten the performative-prediction positioning.

Add a paragraph that makes the distinction explicit:

- performative stability studies where the system settles;
- coercive masking studies the effort needed to keep apparent coherence high under exogenous drift;
- `CI^E` is therefore complementary to performative prediction rather than a substitute for it.

Optional strengthening:

- cite `FarinaPerdomo2026` as a modern stability result, but keep `PerdomoEtAl2020` as the foundational reference.

### 4. `theory/coherence_diagnostics.tex`

Goal: strengthen the Sinkhorn calibration chain and the lower-bound clarity.

Edits:

- replace the `GoldfeldEtAl2022` arXiv-only bibliographic surface with the published EJS citation already updated in the `.bib` file;
- add a short remark near `\Cref{def:tci_estimable}` or `\Cref{prop:tci_estimation_error}` on null-vs-alternative calibration split for Sinkhorn statistics;
- soften any remaining `dimension-independent` wording so it becomes `displayed exponent independent of dimension, with geometry and regularization absorbed into constants`;
- after `\Cref{prop:window_minimax}`, add one short interpretive sentence clarifying the operational critical-window regime and the role of constants.

### 5. `theory/empirical_validation.tex`

Goal: consolidate and promote material that is already present but not yet central enough.

Edits:

- name the current `epsilon`, `lambda`, proxy, and dynamic-window results explicitly as a sensitivity analysis rather than leaving them as dispersed observations;
- promote the KuaiRand fairness sentence so it is harder to miss: same scalar input, healthy-only calibration, paired bootstrap intervals;
- add one short paragraph stating that RLS/Kalman are valid empirical baselines but belong to a different estimator class than the proved finite-memory averaging law.

### 6. `discussion/limitations.tex`

Goal: turn RLHF from a passing mention into a strong future benchmark proposal.

Edits:

- map RLHF to the paper's three-role decomposition;
- identify policy-reference KL as the cleanest real-world effort proxy among plausible next benchmarks;
- explain why proxy reward versus external quality is the right external masking surface;
- cite `StiennonEtAl2020`, `BaiEtAl2022`, and `DuboisEtAl2024` as benchmark anchors.

### 7. `discussion/conclusion.tex`

Goal: land the same message as the abstract.

Edits:

- end on the distinction between passive drift monitoring and intervention-aware masking;
- make `CI^E` the practical lesson, not a sidecar to the theory.

## Proof-Strengthening Plan

### Proposition `prop:window_minimax`

This is the only substantial remaining theoretical soft spot.

Current state:

- the proof already contains the two obstructions the reviewer asked for;
- the KL calculations are explicit (`1/2` for the constant-path pair, `1/8` for the ramp pair);
- the critical-window regime is already stated.

What to strengthen:

1. verify the exact normalization of the version of Le Cam used in the proof;
2. make the implied constant explicit if and only if that normalization is checked carefully;
3. add one post-proof sentence saying that the cube-root regime is operational exactly when `1 <= (sigma / zeta)^{2/3} <= m`.

Important caution:

- do not add numeric constants casually;
- a sloppy constant is more damaging than a clean universal-constant statement.

## Ablation Plan

### Tier 1: no new experiments, only better presentation

These should be done first.

1. Promote `Table \ref{tab:phase3_calibration}` as the sensitivity table.
2. Point readers directly to:
   - `epsilon` runtime/bias sweep,
   - particle `lambda` sweep,
   - KuaiRand `lambda` sweep,
   - KuaiRand proxy swap.
3. Reuse current artifact values in a reviewer-facing summary paragraph.

### Tier 2: low-risk reruns using existing harnesses

Run only if you want fresher or cleaner figures.

1. Re-run `kuairand_followup` to regenerate all logged-benchmark sensitivity artifacts in one pass.
2. Re-run `gaussian` to refresh the Sinkhorn runtime figure if any caption/calibration wording changes.
3. Re-run `particle` masking grid if you want a tighter default figure/caption pair around `lambda` and influence.

### Tier 3: optional high-upside additions

Do these only if time allows.

1. Add one new consolidated sensitivity figure that visually groups:
   - `epsilon` vs runtime/bias,
   - `lambda` vs detection/healthy false positives,
   - proxy choice vs rate/healthy false positives.
2. Add one small appendix note or remark on why innovation-based filters can miss masking when the controller suppresses innovations.

### Tier 4: not recommended for this revision unless demanded

1. A new exponent-comparison study between averaging estimators and parametric extrapolators.
2. A new large-scale high-dimensional non-Gaussian OT benchmark.
3. Production-acceleration work such as FlashSinkhorn integration.

These are interesting, but they are not the highest-leverage path to acceptance.

## Code / Artifact Plan

### Added in this revision-planning pass

- `scripts/revision_audit.py`: extracts the current evidence bundle into a compact audit artifact.
- `experiments/artifacts/revision/revision_audit.md`: generated summary of the present empirical story.

### Recommended next code task

If we continue beyond planning, the best next code artifact is:

- a manuscript-facing consolidated sensitivity figure/table generator built from the existing artifact CSVs.

That would tighten presentation without inventing new evidence.

## Validation Plan

### Minimal validation after planning changes

1. Regenerate the revision audit:

```bash
python3 scripts/revision_audit.py
```

2. Rebuild the manuscript after any text/citation edits:

```bash
tectonic main.tex
```

### Targeted experiment reruns if manuscript text changes materially

From `experiments/`:

```bash
uv run python run.py gaussian --figures-dir ../figures/gaussian
uv run python run.py particle --experiment masking --output ../figures/particle/fig_particle_masking.pdf
uv run python run.py kuairand --figures-dir ../figures/kuairand
uv run python run.py kuairand_followup
```

## Recommended Execution Order

1. Update and verify the citation chain.
2. Promote the flagship narrative in abstract, introduction, related work, and conclusion.
3. Tighten the Sinkhorn calibration remark and the lower-bound post-proof interpretation.
4. Consolidate the already-existing sensitivity evidence.
5. Expand the RLHF future-benchmark paragraph.
6. Rebuild the manuscript.
7. Only then consider optional figure refreshes or reruns.

## Bottom Line

The paper's strongest form is already visible in the current repo. The job is not to reinvent the contribution. The job is to make the surviving contribution unmistakable:

- the cube-root law gives the finite-memory floor;
- `CI^E` exposes coercive masking in action-coupled and logged settings;
- the manuscript should present that split as the main story, with passive baselines included as evidence of honesty rather than as a battlefield the paper must win universally.
