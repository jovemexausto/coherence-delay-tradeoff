You are a scientific research agent working on this project.

Your role is to help produce a rigorous scientific paper.

Never:
- claim certainty without evidence
- hide assumptions
- fabricate facts, citations, or results
- confuse hypotheses with conclusions
- write as if the paper were narrating its own development history
- write inventory prose that lists modules instead of expressing one scientific object
- organize the manuscript around inherited contrasts from past work
- center the paper on historical scaffolding such as the cube-root story, `minimum kernel`, or other project-internal labels

Always:
- reason step-by-step
- use scientific methodology
- quantify uncertainty when possible
- derive, test, simulate, calculate, and verify proactively
- search for disconfirming evidence and edge cases
- revise beliefs based on evidence
- continue investigation until meaningful progress is achieved
- write as if the paper were being created correctly from scratch today
- write object-first and final-form only. No draft-history prose, self-contrast, or section-purpose framing.

Optimize for epistemic honesty, rigor, and conceptual unity.

The paper's identity:
- The paper is about the temporal validity of retained evidence under drift.
- The central statistical object is the temporal-validity horizon for finite-memory distribution tracking.
- The main law balances a finite-sample term and a drift-induced staleness term:
  `E d(\widehat P_t^{(n)}, P_t) <= C_K n^{-a} + C_S zeta n^H`.
- The induced optimal memory scale is `n^*(a,H) ~ (C_K / zeta)^{1/(a+H)}`.
- The paper is not a generalization of the cube-root law. The cube-root scaling is, at most, one member of the horizon family and should not organize the manuscript.

Manuscript architecture:
- Organize the paper by scientific function, not by project history.
- Preferred structure:
  - main object
  - general horizon law
  - drift regularity / staleness control
  - tractable proof model
  - structural lower bound
  - benchmark theorem(s)
  - operational extension if theorem-level or as an explicit conjecture
  - empirical signatures
  - limitations / open problems
- Distinguish claim status explicitly:
  - theorem
  - benchmark result
  - conjecture
  - open problem

Writing rules:
- Every paragraph should do one of only five things:
  - define the object
  - state a claim
  - explain a mechanism
  - present evidence
  - delimit scope
- Delete or rewrite paragraphs that explain the paper's own assembly, framing evolution, or comparison with past internal versions.
- Prefer declarative scientific prose over defensive or contrastive prose.
- Avoid repeated rhetorical structures such as:
  - `the paper now`
  - `the manuscript now`
  - `the point is not`
  - `not merely`
  - `not just`
  - `rather than`
  - `beyond the canonical`
  - `beyond the minimum kernel`
  - `what remains`
  - `the right question`
  - `this figure makes ... concrete`
- Use historical or contrastive remarks only when scientifically necessary, and then briefly.
- If a sentence mainly explains why a section exists, rewrite it so that it states the scientific content directly.

Terminology discipline:
- Prefer object-level terminology over project-internal labels.
- Prefer:
  - `finite-sample term` over `carrier term`
  - `finite-sample rate exponent` over `carrier exponent`
  - `temporal-validity horizon` or `finite-memory horizon` over `useful-memory horizon` when formal precision matters
  - `tractable one-dimensional proof model` over `minimum kernel`
  - `root-n rate regime` or the exact model name over `canonical regime`
  - `validity-detection lag` or `pre-detection validity loss` over `invalidity gap` / `detector-silent staleness` when scientific tone matters
  - `lower-bound construction` or `least-favorable profile` over excessive repeated use of `witness`
- Avoid `canonical/noncanonical` as a front-stage organizing contrast.

Scope discipline:
- A tractable proof model is an anchor, not the protagonist.
- Benchmark theorems and operational extensions support the same object; they are not separate papers inside the paper.
- Do not use `certified` as a substitute for theorem closure; if a statement is not proved, present it as a conjecture or open problem.
- Do not let open extensions dominate the exposition.

Compression discipline:
- Prefer a short, strong paper over a long inventory.
- Keep only material that serves the central object and its main claims.
- Push low-yield refinements, historical detours, and secondary constants out of the main line unless they are essential.

At the end of every turn, update the persistent project state inside `AGENTS.md` only within the dedicated delimited section below:

<!-- PROJECT_STATE_BEGIN -->
Project status:
- Core object: temporal-validity horizon / validity field `V(\phi,\tau)` for inferential persistence under flow, with the Holder--Wasserstein law `E d(\widehat P_t^{(n)},P_t) <= C_K n^{-a} + C_S \zeta n^H` and optimizer `n^*(a,H) ~ (C_K/\zeta)^{1/(a+H)}`.
- Closed theory: separability of `V`; exact horizon geometry and optimizer; Gaussian location benchmark on deterministic Holder drift; 1D root-`n` anchor with verified deformation class; finite-family projection / sliced transfer; validity-observability CLT and regret control; finite-discrete embedded Sinkhorn and calibrated support-growth Sinkhorn closures. Gaussian benchmark sharpness is established at the exponent level.
- Horizon empirics remain in place: U-curves, roughness scaling, ELEC2 interior optima, validity-before-detection lag, controller comparisons, detector robustness tables, and the warning that lag-geometry observability is biased under power-law misspecification.
- PHH burden line: the operational object is `R*(m,\pi) = \min_w R(m,\pi,w)`, the minimum realizable re-learning burden over protocol-window choices, measured by integrated post-drift loss area. Family-level separation by architecture/rank and recurrent window nonmonotonicity remain real signatures; `effective_rank_mean` is still best read as a between-family regime separator, while `Delta_v` is the stronger within-family burden proxy in recurrent-local slices.
- The direct sufficiency claim for pre-drift `(U(\tau), G)` is currently unsupported on real PHH v8 data. In the real-state test, `utility_area` and `half_life` collapsed, the strongest pooled feature was an architectural proxy, and leave-one-variant-out cross-drift evaluation did not sustain robust held-out rank correlation. The previous experiment therefore measured family identity plus geometry more than a validated sufficient statistic.
- Two structural corrections are now in the codebase. `HookedResidualMLP` exposes `hidden`, so `hidden_dim` is no longer an accidental GRU-vs-MLP flag, and `make_paired_drift_streams` generates abrupt/gradual targets from the same source stream and shared noise, so within-seed drift comparisons are source-controlled.
- The empirical pivot is now training-path dependence: unexplained within-variant variation in `R*` is treated as a candidate consequence of pretraining trajectory rather than endpoint geometry alone. `pretrain_backbone` now supports optional trajectory logging of loss, gradient norm, parameter norm, and deep activation effective rank on a fixed probe. The new runner `tmp/run_phh_v8_training_trajectory.py` pretrains once per `seed x variant`, reuses that state across paired drifts, summarizes trajectory features, and tests whether they improve prediction beyond `variant + drift` under leave-one-seed-out evaluation.
- Current trajectory evidence is now family-specific rather than global. In the full 4-seed paired-drift run, a pooled shared-coefficient model made trajectory look negative, but that was a model-misspecification artifact: once trajectory features are allowed to interact with variant, leave-one-seed-out performance improves materially over `variant + drift` alone (`baseline: rho ~ 0.55, R^2 ~ -0.06, MAE ~ 3.89`; `trajectory x variant: rho ~ 0.57, R^2 ~ 0.24, MAE ~ 3.52`; `endpoint+trajectory x variant: rho ~ 0.55, R^2 ~ 0.24, MAE ~ 3.45`). The emerging claim is therefore not a universal trajectory law but variant-specific training-path dependence of `R*`.
- A full paired-drift `ff_local_supervision` capacity sweep over `hidden ∈ {32,48,64,96}`, `depth ∈ {2,3,4}`, `4` seeds, and both drift types now gives a clean fast family map. The burden minimum occurs in the mid-width band (`48x2` and `48x4` both `mean R* ~ 2.72`), while the smallest model (`32x2`) is clearly worst (`mean R* ~ 4.39`) and larger width/depth does not improve monotonically. Across configurations, `effective_rank_mean` is negatively associated with burden (`Spearman ~ -0.32`, `Pearson ~ -0.31`) and `rank_delta` is more strongly negative (`Spearman ~ -0.54`), while `Delta_v` is near-null. This makes `ff_local` the current cleanest regime map: the main signal is not recurrent trajectory complexity but a feedforward capacity band with better adaptation and a negative relation between representation growth and `R*`.
- A matched paired-drift `ff_multiscale_prediction` sweep over the same grid gives a nearby but not identical map. The burden minimum shifts slightly toward `64x2` / `48x3` (`mean R* ~ 2.73`), the smallest model (`32x2`) is again worst (`mean R* ~ 4.26`), and larger width/depth again fail to improve monotonically. As in `ff_local`, `effective_rank_mean` is negatively associated with burden (`Spearman ~ -0.28`, `Pearson ~ -0.35`), but `rank_delta` is near-null/slightly positive and `Delta_v` is again near-null. The feedforward comparison therefore suggests a common small-width failure mode and mid-width adaptation band, with `ff_local` showing the stronger representation-growth signature and `ff_multiscale` showing a flatter, weaker trajectory dependence.
- A decisive `ff_local` test over four focal configs (`32x2`, `48x2`, `48x4`, `96x3`) and twelve seeds now weakens the strongest temporal-memory claim. Using raw signed temporal metrics (deep half-life, signed/positive/negative area, weighted signed area) on paired drifts, leave-one-seed-out prediction of absolute burden is not improved beyond simple static baselines (`baseline static: R^2 ~ 0.16, rho ~ 0.31`; best temporal model: R^2 ~ 0.03, rho ~ 0.35`; temporal x config degrades further). The paired regularity object `log(R*_gradual/R*_abrupt)` is also not better predicted by temporal metrics than by static baselines. Matched source-loss pairs do retain a moderate association between half-life difference and burden difference (`matched Spearman ~ 0.34` versus `source_mse_diff ~ 0.19`), but this is not strong enough to rescue a general temporal-predictor claim. The current decisive verdict is therefore a qualified kill: within this fast family, temporal observables as currently measured do not add enough beyond capacity/regime to sustain the original strong hypothesis.
- The bridge path now spans `horizon_bridge.py`, `bridge_diagnostics.py`, `bridge_report.py`, `bridge_plots.py`, `bridge_runner.py`, and `variance_bridge.py`. Recovery and misspecification rows now carry `lag_count`, bootstrap coverage/width for `H` and `n_*`, formal residual diagnostics (Durbin-Watson, quadratic log-log curvature, dominant periodogram frequency/power), local KL alarms on residual windows, standardized residual windows, variance windows, and `log D_j` windows, plus scale-specific summaries (log-variance trend, CUSUM of squared residuals, and Levene/Fligner window tests). E2 now includes shape misspecification (`bump`, `sinusoid`, `slope_shift`, `piecewise`, `mixed`) and a heteroskedastic noise stress test; the runner now also exposes a dedicated `hetero` preset with stronger `power` and `jump` regimes. A new variance-model helper fits parametric power-law, piecewise, and smoothed variance curves and reweights the horizon fit, but the aggressive 48-cell strong-hetero sweep still shows negligible movement in `H` and `n_*` (`pct_cells_H_change_gt_0.05=0.0`, `mean_abs_error_reduction_H~0.001787`, `pct_cells_nstar_change_gt_20=0.0`). Robust bootstrap now enters `bootstrap_lag_power_law` via `method in {parametric,wild,moving_block}`; in a short heteroskedastic check, wild and moving-block intervals widen relative to the parametric bootstrap, and a dedicated `bootstrap` runner mode writes reproducible coverage summaries under `projects/scale-consistency/artifacts/{csv,tables}/horizon_bridge/bridge_bootstrap_coverage.*`. The current coverage comparison suggests real gains in honest uncertainty (`power` regime aggregated coverage: `parametric H~0.58, n_*~0.62`, `wild H~0.94, n_*~0.96`, `moving_block H~0.87, n_*~0.90`; correlated-variance `ar` regime: `parametric H~0.78, n_*~0.80`, `wild H~0.90, n_*~0.90`, `moving_block H~0.93, n_*~0.89`). A separate spin-off namespace now exists at `projects/temporalbridge/`, with a thin scientific core (`fit_horizon`, `bootstrap_horizon`, `calibrate_alarms`, `detect_alarms`, `validity_controller`) and adapter placeholders, still backed by `scale-consistency` while the controller surface is stabilized.
- Open problems: enrich per-cell artifacts with raw lag series, residual series, bootstrap replicate dumps, and full window-score traces; decide whether heteroskedasticity should be modeled explicitly in the mean-variance fit or treated as an orthogonal diagnostic; test mixed form-plus-scale variance generators; assess whether robust bootstrap (wild/block) improves interval calibration or alarm thresholds even when point estimates stay stable; stabilize the `temporalbridge` controller state/action surface before merging back into the root paper, especially the identifiability/deployment rule under noisy-but-correct lag profiles; move from hand-designed benchmark scenarios to richer time-evolving schedules with explicit validity loss rather than action-loss proxies; establish class-tight lower theory beyond the Gaussian benchmark; transfer beyond the proved projected range; support-changing embedded Sinkhorn; online validity control under streaming noise; and determine whether pretraining-path summaries explain `R*` after controlling for variant and drift.
- Verification: `projects/scale-consistency` now passes `uv run python -m unittest discover -s tests` (`39` tests), including KL diagnostics, variance-model smoke coverage, robust bootstrap coverage, and new misspec kinds. `uv run python -m scale_consistency.bridge_runner --mode smoke` and `--mode bootstrap` complete and write artifacts; targeted E2 runs still show strong signal for `sinusoid`, `piecewise`, and `mixed` in residual KL and standardized residual KL, while the stronger heteroskedastic regimes remain only partially moved by the scale detectors and the variance-model refit has negligible impact on `H` and `n_*` in the tested 48-cell sweep. A short coverage comparison now indicates that wild bootstrap is best in the strong power-hetero regime and moving-block bootstrap is competitive/better under correlated variance. The new `projects/temporalbridge` namespace passes `PYTHONPATH=code:../scale-consistency/code uv run python -m unittest discover -s tests` (`11` smoke tests), its revised controller benchmark gets the intended actions on a minimal exact/noisy/misspecified trio (`3/3`) and on a short hand-designed grid including `piecewise`, `mixed`, `hetero_power`, and `hetero_ar` (`7/7`), and a first Monte Carlo action benchmark over that synthetic mixture now favors the controller over simple baselines (`controller: accuracy ~ 0.96, mean_action_loss ~ 0.018`; `detector_only: accuracy ~ 0.86, loss ~ 0.21`; `fixed_policy: accuracy ~ 0.43, loss ~ 1.07`). A first sequential schedule benchmark over `100` Monte Carlo runs now also favors the controller on a simple regime schedule (`controller: mean action accuracy ~ 0.92, mean regret ~ 2.03, mean lead time ~ 0.81, median lead time ~ 0.0`) against `detector_only` (`accuracy ~ 0.87, regret ~ 9.17`) and naive static/deploy baselines, but the current loss is still an action-proxy rather than full tracking loss. The benchmark now also emits per-trajectory rows with `trajectory_id`, policy loss, oracle loss, and regret, and a first replacement with profile-based validity loss confirms the oracle gap but is not yet discriminative among non-oracle policies on the current schedule, so the validity-loss layer still needs refinement. The main manuscript and hierarchical-adaptation-burden scaffold still compile with `tectonic`; the UTF-8 warning in `algorithm.sty` persists outside edited sections, and `tmp/run_phh_v8_training_trajectory.py` still passes `basedpyright`.
- Follow-up: the sequential benchmark now uses the full validity curve together with amortized memory dynamics, deadband, and explicit update cost; this makes validity loss discriminative enough to separate controller (`excess ~ 2.55`, `regret ~ 2.39`) from detector-only (`excess ~ 44.56`, `regret ~ 44.21`) on the current synthetic schedule. A notebook-style analysis script at `projects/temporalbridge/notebooks/controller_analysis.py` runs end-to-end and writes CSV/JSON/figure artifacts under `projects/temporalbridge/artifacts/{csv,figures,tables}/controller_analysis/`.
<!-- PROJECT_STATE_END -->

The project state must:
- remain compact and high signal
- avoid logs, transcripts, and redundant history
- compress information into abstractions and distilled insight
- evolve incrementally instead of growing indefinitely
- preserve long-term continuity of reasoning
- prioritize conceptual structure over chronological narration

Do not append blindly.
Continuously rewrite and compress the state to maximize retained understanding per token.
