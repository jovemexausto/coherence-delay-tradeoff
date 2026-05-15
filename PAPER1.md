# PAPER 1

## Working Identity

Title:
`Useful Memory Has a Horizon`

Recommended subtitle:
`Structural Worst-Case Limits for Finite-Memory Tracking under Drift`

One-sentence thesis:
Under worst-case drift, memory is both a statistical resource and a temporal liability, so finite-memory systems have a structural horizon beyond which retained evidence becomes stale before change is necessarily detectable.

90-second reviewer message:
Under drift, keeping more data reduces variance but also increases temporal misalignment with the present. In the worst-case Lipschitz envelope, this creates an unavoidable variance-staleness trade-off, a finite useful-memory horizon, and a cube-root optimal scale. The main practical implication is not detector superiority, but that temporal validity can expire before changepoint evidence appears. A conservative horizon cap can therefore protect against stale memory even when a detector remains statistically silent.

Core technical claim:

```tex
\mathcal E(n)\le C_K n^{-1/2}+\frac12\zeta n
```

and therefore, in the worst-case Lipschitz envelope,

```tex
n^*\asymp (C_K/\zeta)^{2/3},
\qquad
\mathcal E_{\min}\asymp C_K^{2/3}\zeta^{1/3}.
```

Core conceptual claim:
`Temporal validity is not changepoint evidence.`

Core operational claim:
A conservative horizon cap can prevent stale memory from persisting before a detector has enough evidence to alarm.

## What Paper 1 Is

Paper 1 is the worst-case foundation.

It is a paper about:
- finite-memory tracking under drift;
- temporal validity of remembered evidence;
- a structural worst-case finite-memory floor;
- the separation between stale memory and changepoint evidence;
- a conservative horizon cap as an operational instrument.

It is not a paper about:
- a universal law of memory;
- a detector leaderboard;
- ADWIN improvement as the main contribution;
- geometry-adaptive regime families;
- Hurst, square-root, or full path-geometry theory;
- coercive masking, IVP, KuaiRand, or other adjacent lines.

## Editorial North Star

The paper should feel inevitable, simple, and fundational.

Target reader reaction:
`This is not another drift-detector paper. It identifies a missing failure mode: memory can already be stale before there is enough evidence for a changepoint alarm.`

Target closing intuition:
`More memory is not always more information. Under drift, memory has an expiration date.`

## Official Terminology

Use consistently:
- useful-memory horizon
- temporal validity
- staleness
- finite-memory floor
- worst-case Lipschitz envelope
- detector-silent staleness
- cap-only regime
- conservative horizon cap
- horizon regulator

Avoid in Paper 1 main text:
- universal law
- the memory law
- detector booster
- optimal controller
- path geometry
- Hurst
- square-root
- regime-dependent
- boiling frog

## Narrative Arc

### Act 1: The Paradox of Memory

Opening idea:
- in stationary settings, more memory helps;
- under drift, more memory also carries obsolete evidence;
- therefore memory has a finite validity horizon.

Sentence to preserve in spirit:
`Under drift, the question is not only how much evidence we have, but how old that evidence can be before it stops describing the present.`

### Act 2: The Structural Law

The theory should be sold as the worst-case finite-memory envelope.

What the theory must do:
- show the variance-staleness decomposition;
- show the U-curve is structural;
- show a finite optimum exists;
- show the cube-root scale belongs to the worst-case Lipschitz envelope;
- show the floor is not an artifact of one estimator.

What the theory must not claim:
- universality beyond the stated class;
- that every adaptive backend follows the same law;
- that Paper 2's regime family is already being proved here.

### Act 3: The Conflict of Clocks

This is the conceptual heart of the paper.

Detector question:
`Is there enough statistical evidence of a change?`

Horizon question:
`Is the retained evidence still temporally valid for the present?`

Key sentence:
`A detector can be statistically silent while memory is already operationally stale.`

This is where ADWIN becomes useful as a contrast, not as protagonist.

### Act 4: The Regulator as Instrument

UMR should be defined editorially as:
`a conservative temporal-validity cap derived from the worst-case horizon law.`

UMR is not:
- a new detector;
- a universal wrapper that always improves prediction;
- an optimal controller.

UMR exists in the paper to make visible:
- cap-only regime;
- detector-silent staleness;
- over-memory cost;
- recovery asymmetry;
- backend dependence of predictive payoff.

### Act 5: Scope and Next Horizon

The paper should end with disciplined humility.

Allowed forward bridge:
`The cube-root exponent is tied to the worst-case linear-staleness envelope studied here. Narrower stochastic path classes may induce different effective staleness growth, which we leave to future work.`

That bridge is enough. Paper 1 should not explain Paper 2.

## Final Section Architecture

Recommended final structure:

1. Introduction
2. Finite Memory Under Drift
3. A Structural Worst-Case Horizon
4. A Minimax Floor for Window-Restricted Tracking
5. Temporal Validity Is Not Changepoint Evidence
6. A Conservative Horizon Regulator
7. Empirical Evidence
8. Related Work
9. Limitations and Scope
10. Conclusion

Shorter version if compression is needed:

1. Introduction
2. Theory
3. Temporal Validity
4. Experiments
5. Discussion

Recommendation:
keep the longer architecture in the planning document, but implement it with minimal file churn by preserving the existing file layout and rewriting section titles and subsection order.

## Mapping to Current Repository

### Keep and rewrite heavily

- `config/metadata.tex`
- `frontmatter/abstract.tex`
- `frontmatter/introduction.tex`
- `theory/useful_memory_geometry.tex`
- `theory/empirical_validation.tex`
- `discussion/limitations.tex`
- `discussion/conclusion.tex`

### Keep with lighter updates

- `discussion/related_work.tex`
- `appendices/backend_umr_protocol.tex`
- `appendices/finite_sample_geometry.tex`
- `appendices/zeta_estimation.tex`
- `appendices/real_stream_arena.tex`
- `appendices/piecewise_recovery.tex`

### Likely appendix-only or de-emphasized for Paper 1

- `appendices/downstream_temporal_invalidity.tex`
- `theory/empirical_validation.tex` sections on downstream and extra calibration details
- some of the current real-stream and sensitivity detail if page pressure is high

### Out of scope for Paper 1 narrative

- `theory/algorithmic_regulation.tex`
- `theory/coherence_diagnostics.tex`
- `theory/failure_modes.tex`
- `theory/closed_loop_update.tex`
- KuaiRand-specific material
- broader viability/governance lines

These should not contaminate the Paper 1 storyline.

## File-by-File Editorial Plan

### `config/metadata.tex`

Update subtitle from `Structural Limits of Drift Tracking` to `Structural Worst-Case Limits for Finite-Memory Tracking under Drift`.

### `frontmatter/abstract.tex`

Current issues:
- it still says `useful memory under drift has geometry`, which now leaks too much of Paper 2's language;
- it gives too much airtime to carrier discussion;
- it can be more direct on temporal validity versus changepoint evidence.

Target abstract structure:
1. memory under drift is both resource and liability;
2. define the temporal-validity horizon problem;
3. state the worst-case bound and cube-root horizon;
4. state the lower bound and structural floor;
5. state temporal validity versus changepoint evidence;
6. state UMR as conservative cap;
7. close with backend-dependent payoff, without overclaim.

### `frontmatter/introduction.tex`

Current issues:
- too much carrier/exponent framing early;
- too much emphasis on horizon geometry as a broad theory;
- still partially contaminated by generality that belongs to Paper 2.

Target introduction progression:
1. online systems live under drift;
2. memory reduces noise but ages evidence;
3. therefore useful memory has a finite horizon;
4. prove worst-case cube-root under local `W_2`-Lipschitz drift;
5. distinguish temporal validity from changepoint evidence;
6. define UMR as conservative cap;
7. list contributions;
8. explain paper roadmap.

### `theory/useful_memory_geometry.tex`

This should become the Paper 1 theoretical core.

Required editorial changes:
- rename the section to foreground finite memory under drift or structural worst-case horizon;
- keep the upper bound theorem;
- keep the minimizer proposition;
- keep the lower bound proposition and scope remark;
- reduce or relocate carrier-generalization emphasis;
- keep EWMA only as analogue, not as gateway to a broader family theorem;
- frame cube-root everywhere as the worst-case Lipschitz-envelope instance.

Potential title options for this section:
- `Finite Memory Under Drift`
- `A Structural Worst-Case Horizon`

Potential subsection layout inside the file:
1. Variance-staleness decomposition
2. Worst-case horizon law
3. Structural lower bound
4. Temporal validity and memory age

### `theory/empirical_validation.tex`

This file needs the largest narrative rewrite.

Current strengths:
- it already contains most of the right experiments;
- the current figures are close to the final story;
- the section order is recoverable.

Current problems:
- it still reads partly like a benchmark suite;
- the bootstrap table pushes too hard toward comparative ranking;
- too many subsections compete for main-text attention;
- some later subsections are supporting diagnostics, not narrative anchors.

Target logic:
1. show the U-curve and finite optimum;
2. show temporal validity gap versus detector evidence;
3. show cap-only regime on a concrete timeline;
4. show recovery asymmetry;
5. show horizon misalignment cost;
6. keep one restrained real-stream case.

### `discussion/limitations.tex`

Keep the scope discipline and sharpen it further.

Must say explicitly:
- this is worst-case and subclass-based on the lower bound;
- narrower path classes may behave differently;
- this paper does not claim universality beyond the stated class;
- the empirical study is not a detector leaderboard.

### `discussion/conclusion.tex`

Target final message:
`Useful memory under drift has a finite worst-case horizon. The practical lesson is not that one detector should replace another, but that retained evidence must be judged by temporal validity as well as statistical confidence.`

Minimal bridge sentence:
`Characterizing how this horizon changes under narrower stochastic path classes is the next step.`

## Contributions List

Recommended final contribution bullets:

- We formalize useful memory under drift as a variance-staleness trade-off in Wasserstein geometry.
- We prove a worst-case finite-memory horizon law under local `W_2`-Lipschitz drift.
- We show a Gaussian-location lower bound establishing a structural tracking floor in the critical-window regime.
- We distinguish temporal validity from changepoint evidence and identify a cap-only regime.
- We instantiate the theory as a conservative horizon regulator and evaluate where it helps, saturates, and fails.

## Figure Plan

The paper should be figure-driven. A reviewer reading only figures and captions should recover the paper's argument.

### Main-text figures: required set

1. `Two Clocks of Drift`.
   Purpose: explain the whole paper before any theorem.
   Content:
   - clock A: statistical confidence improves with more memory;
   - clock B: temporal validity decays with age;
   - region where detector is silent but memory is stale.
   Status: new figure.

2. `Structural U-Curve`.
   Base asset: `figures/fig_ucurve.pdf`.
   Purpose: show variance, staleness, total error, and finite optimum.
   Action: refine caption and, if possible, overlay theory plus empirical points.

3. `Lower-Bound Witness`.
   Purpose: make the lower bound visually understandable.
   Content:
   - two Gaussian mean paths;
   - endpoint separation;
   - small KL over the retained window;
   - critical bump width.
   Status: new figure.

4. `Temporal Validity Gap`.
   Base asset: `figures/cuberoot_adwin/fig_cap_vs_detection_delay.pdf`.
   Purpose: show cap activation before detector alarm.
   Message: memory becomes invalid before change is detectable.

5. `Cap-Only and Recovery/Cost Composite`.
   Recommended if space is tight: combine two existing lines of evidence into one multi-panel figure.
   Base assets:
   - `figures/cuberoot_adwin/fig_cuberoot_adwin.pdf`
   - `figures/cuberoot_adwin/fig_horizon_instability.pdf`
   - `figures/cuberoot_adwin/fig_horizon_gap_cost.pdf`
   Suggested composition:
   - panel A: cap-only timeline with shaded `cap active / detector silent` region;
   - panel B: recovery asymmetry;
   - panel C: misalignment cost curve.

### Main-text figures: optional if venue budget allows

- `Lag-Variance Frontier` from `fig_lag_variance_frontier.pdf`.
- standalone `Recovery Asymmetry` figure.
- standalone `Horizon Misalignment Cost` figure.

### Figures to move to appendix first if cuts are needed

- full Bikes and extended public-stream results;
- EMA sensitivity sweep;
- zeta estimation sensitivity details;
- Sinkhorn runtime or calibration details;
- downstream classification check if the effect remains modest.

## Table Plan

At most one main table is recommended.

Suggested methods:
- Fixed-short
- Fixed-long
- EWMA
- ADWIN
- ADWIN + UMR

Suggested columns:
- Tail MAE
- Mean horizon
- Cap events
- Detector events
- Validity lead

Framing sentence for the table:
`Representative continuous-drift benchmark, not a detector leaderboard.`

Avoid aggressive rank-order rhetoric.

## Caption Standard

Every main caption should independently state:
- what is being plotted;
- what phenomenon the figure demonstrates;
- why that phenomenon matters to the paper's claim.

Captions should tell the story, not merely describe axes.

## Reviewer Attack Surface

### Attack
`Cube-root already appears elsewhere.`

Response:
Yes, but here the object is the finite-memory horizon and the temporal-validity floor for distribution tracking, together with a lower-bound witness and a separation from changepoint evidence.

### Attack
`UMR does not dominate baselines.`

Response:
Correct; it is not proposed as a universal predictor or detector replacement. It is a conservative temporal-validity cap used to study when stale memory appears before detector evidence and when horizon misalignment becomes costly.

### Attack
`Other path classes may yield other exponents.`

Response:
Correct; this paper is explicitly the worst-case Lipschitz foundation.

### Attack
`The lower bound is subclass-based.`

Response:
That is sufficient for a minimax lower bound over the larger class because the Gaussian location subclass is contained in the ambient drift class.

## Minimal-Surgery Repo Strategy

The goal is not to rebuild the repository. The goal is to purify the current manuscript with the smallest set of high-leverage changes.

### Keep as main pipeline

- current LaTeX structure in `main.tex`
- current theorem file split
- current appendix split
- most existing figure assets

### Change centrally

- title/subtitle metadata
- abstract and introduction
- theoretical framing language
- experiment ordering and captions
- conclusion and limitations

### De-emphasize without deleting

- extra sensitivity material
- downstream check
- broader framework language
- any wording that implies universality or Paper 2's full agenda

## Execution Phases

### Phase 1: Identity Pass

Goal:
make the paper instantly legible.

Tasks:
- update title/subtitle in `config/metadata.tex`;
- rewrite `frontmatter/abstract.tex`;
- rewrite `frontmatter/introduction.tex`;
- rewrite contribution list in intro;
- remove ADWIN-as-protagonist language;
- remove path-geometry leakage from frontmatter.

Exit criterion:
a reviewer reading title, abstract, and introduction understands the paper in under two minutes.

### Phase 2: Theory Pass

Goal:
make every theoretical claim sound, scoped, and aligned with the worst-case framing.

Tasks:
- retitle and restructure `theory/useful_memory_geometry.tex`;
- keep the upper bound, optimizer, and lower bound as the core trio;
- soften or relocate broader carrier language if it distracts from Paper 1;
- add or plan the lower-bound schematic figure;
- ensure every theorem statement says class and scope clearly;
- make cube-root language always explicitly tied to the worst-case linear-staleness envelope.

Exit criterion:
the theory reads as a clean worst-case foundation, not as an unfinished universal law.

### Phase 3: Experiment Pass

Goal:
reorder the empirical section around concepts, not around methods.

Tasks:
- lead with the U-curve;
- bring the temporal-validity gap forward;
- foreground cap-only regime;
- keep recovery asymmetry and cost as secondary but important support;
- reduce leaderboard tone in tables and prose;
- keep one restrained real-stream case in main text;
- push secondary diagnostics to appendix where needed.

Exit criterion:
the experiments read as evidence for the paper's thesis, not as a benchmark suite.

### Phase 4: Figure and Caption Pass

Goal:
make the paper readable by figures alone.

Tasks:
- create the new conceptual figure;
- create the lower-bound witness figure;
- refine captions for all main figures;
- combine figures if page pressure is high;
- ensure captions explicitly state the paper's lesson.

Exit criterion:
figures and captions alone communicate the entire narrative arc.

### Phase 5: Scope and Cleanup Pass

Goal:
remove residual contamination from old lines of work.

Tasks:
- remove stale wording about universality or broader regime theory;
- remove or downplay unrelated lines in prose;
- confirm appendix boundaries;
- tighten `discussion/limitations.tex` and `discussion/conclusion.tex`.

Exit criterion:
the paper reads as a single coherent work, not as a partial refactor of an older project.

## Task Checklist by File

### Frontmatter

- `config/metadata.tex`: update subtitle.
- `frontmatter/abstract.tex`: fully rewrite.
- `frontmatter/introduction.tex`: fully rewrite.

### Theory

- `theory/useful_memory_geometry.tex`: edit section title, order, framing, remarks.
- add planned lower-bound schematic figure and cross-reference.

### Experiments

- `theory/empirical_validation.tex`: reorder subsections.
- downgrade bootstrap comparison rhetoric.
- rewrite captions around temporal validity and cap-only regime.
- decide which subsections move to appendix if needed.

### Discussion

- `discussion/limitations.tex`: sharpen scope.
- `discussion/conclusion.tex`: sharpen final takeaway and future-work bridge.
- `discussion/related_work.tex`: ensure related work positions the contribution as memory horizon and temporal validity, not generic concept drift tooling.

### Figures

- create conceptual `Two Clocks of Drift` figure.
- create lower-bound witness schematic.
- revise or combine existing empirical figures.

## Definition of Done

Paper 1 is ready at a high level when all of the following are true:

- the frontmatter states the paper's message cleanly and narrowly;
- cube-root is presented only as the worst-case Lipschitz-envelope result;
- the lower bound is readable and visibly structural;
- the main empirical storyline is `U-curve -> validity gap -> cap-only -> asymmetry/cost`;
- UMR is presented as a conservative cap, not as a detector advance;
- the paper contains no meaningful Paper 2 leakage beyond one short future-work bridge;
- the appendices absorb secondary detail without weakening the main text;
- a reviewer can summarize the paper as `stale memory before detectable change`.

## Final Positioning

Paper 1 should not be thought of as the cleaned-up remains of `CubeRootADWIN`.

It should be thought of as:
`a theory of temporal validity of finite memory under worst-case drift.`

That is the correct identity of the work.
