# Revision Plan

## Objective

Turn the current manuscript into a standalone paper on the temporal geometry of
useful memory under drift.

This revision is not a light trim of the previous draft. The paper must have:

- its own problem statement;
- its own narrative spine;
- its own experiment and ablation package;
- no conceptual dependence on the coercive-masking branch.

## Final Paper Stance

The paper should read as follows:

- agents with finite memory must track a drifting target distribution;
- larger windows reduce statistical noise but increase informational staleness;
- this induces a cube-root optimal memory law;
- a minimax lower bound shows an unavoidable finite-memory floor;
- freshness / Age-of-Information provides an interpretive lens rather than the
  main theorem.

## What Stays

- finite-memory tracking under Wasserstein drift;
- the cube-root window law;
- the finite-memory floor and lower bound;
- the operational horizon-regulation consequence of the law;
- Gaussian and related tracking experiments that directly support the law;
- EWMA or similar finite-memory comparators;
- a narrow related-work bridge to dynamic regret, nonstationary estimation, and
  AoI/freshness;
- `CubeRootADWIN` as the first operational consequence of the law.

## What Goes

- coercive masking as a central theme;
- `CI^E` as a flagship result;
- RLHF / Goodhart as manuscript framing;
- intervention-aware diagnosis as a primary contribution;
- KuaiRand and particle masking as central evidence;
- any claim that passive-detector comparisons establish universal superiority.
- any framing that makes drift detection the central object instead of memory
  validity.

## Narrative Spine

The manuscript should follow this argumentative order:

1. Pose the tracking problem under nonstationary drift.
2. Explain the finite-memory trade-off between estimation error and staleness.
3. State the cube-root law and the resulting error floor.
4. Show the lower bound for the restricted estimator class.
5. Interpret the result as an information-freshness law.
6. Validate the law with experiments and ablations designed for this story.
7. Show that the same law can be turned into a usable horizon regulator.

## File-by-File Edit Plan

### 1. `frontmatter/abstract.tex`

Goal: make the abstract theorem-first, self-contained, and free of the removed
branch.

Edits:

- open with finite-memory tracking under drift;
- state the cube-root law;
- state the finite-memory floor / lower bound;
- mention AoI/freshness only as interpretation if it helps;
- close with the operational horizon-regulation consequence;
- remove masking, RLHF, and logged-intervention language.

### 2. `frontmatter/introduction.tex`

Goal: give the paper its own identity.

Edits:

- motivate the tracking problem rather than coherence diagnostics;
- explain the variance-vs-staleness trade-off early;
- frame the paper around adaptive memory horizons and useful-memory geometry;
- add a narrow contribution list aligned with the retained claims;
- remove any suggestion that this paper's value depends on the cut branch.

### 3. Theory core files

Primary targets:

- `theory/mathematical_definitions.tex`
- `theory/coherence_diagnostics.tex`
- any theorem file still carrying the retained law and lower bound

Goal: leave only the formal backbone needed for the tracking paper.

Edits:

- keep only the setup needed for drifting-distribution tracking;
- retain the estimator analysis and lower bound;
- keep the debiased Sinkhorn discussion where it supports estimability;
- fix any surviving proposition sign / notation mismatch that affects the
  retained theory;
- remove triadic, ISS, masking, and closed-loop architectural theory from the
  core argument;
- keep the formal path aligned with the operational memory-horizon story.

### 4. `theory/empirical_validation.tex`

Goal: redesign the section around the claims of the new paper.

Edits:

- center the Gaussian sweep / `U`-curve;
- make drift-strength and window-length ablations explicit;
- retain EWMA and any fair finite-memory comparator;
- keep dynamic `n_t^*` illustrations only if they directly support the law;
- keep Sinkhorn runtime/calibration only insofar as it supports the measurement
  story;
- remove masking-centric and logged-intervention results from the main paper.
- add the `CubeRootADWIN` continuous-drift benchmark and its piecewise-oracle
  horizon recovery as the operational consequence of the law.

### 5. `discussion/related_work.tex`

Goal: position the paper next to the right literature.

Edits:

- anchor on `Yang et al. (2016)`;
- add the small dynamic-regret / variation-budget cluster;
- include `Keehan, Anderson & Wiesemann (2025)` as near-setting prior art;
- keep `Genevay et al. (2019)` and `Niles-Weed & Bach (2019)` for the OT side;
- use AoI as an interpretive bridge, not as the main mathematical ancestor;
- keep the framing adjacent to adaptive memory and nonstationary tracking, not
  to drift detection as the primary object.

### 6. `discussion/conclusion.tex`

Goal: land on the theorem and its interpretation.

Edits:

- restate the cube-root law;
- restate the finite-memory floor;
- restate the freshness interpretation;
- restate the operational horizon-regulation consequence;
- mention downstream implications only briefly and without reopening the cut
  branch.

## Empirical Mandate

The paper needs experiments that justify its own claims. The minimum package is:

1. a clear `U`-curve showing the finite-memory trade-off;
2. an ablation over drift strength showing how the best window changes;
3. a comparison with EWMA or another fair finite-memory baseline;
4. an illustration of dynamic `n_t^*` when drift varies over time, if supported
   by the harness;
5. at least one figure or table making the finite-memory floor operationally
   visible;
6. a `cap-only` regime figure or table showing staleness before changepoint
   evidence;
7. only passive benchmark evidence that genuinely supports the tracking story.

If an experiment does not support one of these claims, it should not remain in
the body of `Paper 1`.

## Validation Plan

1. Rebuild the manuscript after each substantial text pass.
2. Rerun only the experiments whose figures or claims survive the rewrite.
3. Verify that every remaining table and figure supports the revised story.
4. Remove stale references to masking, `CI^E`, RLHF, Goodhart, and logged
   interventions.

## Execution Order

1. Lock scope.
2. Rewrite abstract and introduction.
3. Trim and repair the theory core.
4. Redesign the empirics section around the retained claims.
5. Rewrite related work and conclusion.
6. Rebuild and validate.

## Bottom Line

The job is no longer to rescue the old manuscript as a hybrid paper. The job is
to turn the surviving tracking contribution into a focused standalone paper on
useful-memory geometry under drift, with its own narrative and evidence.
