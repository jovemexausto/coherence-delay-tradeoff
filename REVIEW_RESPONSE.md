# Review Response Report

Historical note:

- This report preserves an earlier review-response trail, including branches that are no longer central to the current manuscript.
- The active paper is now the standalone drift paper `Useful Memory Has a Horizon: A Cube-Root Law for Tracking Under Drift`.

## Purpose

This report is the shared reference for follow-up phases. It records what the review got right, what must be corrected in the paper, what evidence already exists, and what still needs new proofs, ablations, or baselines.

## Executive Decision

We will pursue the strongest defensible version of the contribution, even if that requires new proofs and new ablations. The paper must reflect evidence, not defend a claim after the evidence weakens it.

The project remains centered on the finite-memory floor and the operational distinction between temporal validity and changepoint evidence.

## Current Stance

- finite-memory tracking under drift has a real finite-memory floor;
- the current manuscript is theorem-first, not detector-first;
- archival masking / intervention-aware materials remain in the repo, but are not the core of the present paper.

## Reviewer Questions and Responses

### 1. Can you formalize the upper bound?

Yes. The manuscript now states a formal theorem in `theory/useful_memory_geometry.tex`
(`\cref{thm:lag_error}`) for the uniform-window empirical estimator under a
`W_2`-Lipschitz drift assumption. The theorem is written as an explicit bound,
`\mathcal{E}(n) \le C_K n^{-1/2} + \tfrac{1}{2}\zeta n`, with a proof sketch
that splits the statistical term from the staleness term via a transport
decomposition.

### 2. Can you correct Proposition 2.7?

Yes. The proposition now matches the proof: the lower bound is stated as
`\max(c_0 m^{-1/2}, c_1 \sigma^{2/3}\zeta^{1/3})`, not as a linear-in-`m`
staleness term. This is the right critical-window conclusion supported by the
constant-path and ramp constructions.

### 3. How is `\zeta` estimated online, and how sensitive is the result?

The estimator is now spelled out in `appendices/zeta_estimation.tex`. The drift
proxy is a short-window block difference, smoothed with an EMA, and the appendix
includes the empirical `\alpha` sweep already used in the code. The key result is
that the contraction/expansion asymmetry persists across the tested range and the
expansion-to-contraction ratio stays above one.
The appendix also records a prefix-validated calibration study for `(d,\alpha)`;
it is treated as an operational refinement and sensitivity analysis, not as a
replacement for the fixed-regulator core results.

### 4. What is the evaluation metric, and how is `\varepsilon` chosen?

The theory is stated in `W_2`. The Sinkhorn divergence appears only as the
computational realization of the metric layer in the synthetic experiments. The
main text now says this explicitly, and the runtime experiment varies `\varepsilon`
over `{0.02, 0.05, 0.2, 1.0}` to show sensitivity. The new finite-sample appendix
also quantifies how `\varepsilon` and sample size inflate the effective horizon
calibration constant; the paper does not claim that the main cube-root exponent
depends on a special `\varepsilon` choice.

### 5. Can you add stronger adaptive baselines?

Yes. The real-stream benchmarks now use a backend+UMR arena: `EWMA`,
Window-Dilemma-style switching, MELO-style multi-horizon hedging, and `ADWIN`
are each evaluated with and without the regulator, alongside `RLS`, `Kalman`,
and fixed-memory references. This makes the practical comparison more direct:
the cube-root regulator is no longer only compared to detectors and fixed
memory, but also to explicit horizon-selection and expert-mixing baselines.

### 6. Can you provide a real-data case study?

Yes. The manuscript now includes `ELEC2` and `Bikes` as real-stream robustness
checks for the backend+UMR arena. These sections show that the same
temporal-validity gap appears on public demand streams, while ADWIN remains the
stronger comparison baseline among the tested warning methods.

## Validation Notes

- The `zeta` sweep CSV matches the appendix table.
- The cube-root minimizer still checks numerically: `n* = (C_K/\zeta)^{2/3}`
  reproduces the stated minimum error exactly up to floating-point precision.
- The real-stream summaries for ELEC2 and Bikes were regenerated from the current
  code.
