# 13. Scale-Consistency V1 Plan
Status: active
Category: task
Prev: 10. Paper Next Steps

## Central object

The spin-off paper studies structural inference for noisy lagged transport
discrepancies that admit a low-dimensional power-law geometry

`D_j = zeta j^H`, `j = 1, ..., L`.

The statistical difficulty is that the observed quantities are estimates
`\hat D_j`, and the leading log-domain variance depends on the same scale law
that is being inferred.

## V1 paper contract

The V1 paper should do only six things:

1. define the noisy scale-consistency model
2. derive feasible inference for the exponent and scale law
3. state and prove residual adequacy calibration
4. state and prove the minimax lower bound for model distinguishability
5. state and prove adaptive attainment of the minimax rate
6. present numerical evidence directly tied to those claims

The paper is successful when a reader can identify without ambiguity:

1. the statistical object
2. the null and alternative hypotheses
3. which results are theorem-level
4. what the simulations are evidence for
5. which extensions are deliberately left open

## V1 section architecture

1. Introduction
2. Model and hypotheses
3. Feasible log-domain estimation
4. Residual adequacy of the power-law family
5. Minimax lower bound
6. Adaptive attainment
7. Numerical evidence
8. Operational implication
9. Discussion and open problems

## Claim ledger

### Theorem

1. FWLS central limit theorem in the fixed-`L` heteroscedastic log-domain model
2. Exact oracle `chi^2(L-2)` null law and asymptotic transfer to feasible weights
3. Minimax lower bound for distinguishing the power-law family from alternatives with `kappa >= kappa*`
4. Matching upper bound for the constructed test
5. Adaptive attainment of the minimax rate without prior knowledge of `H`

### Benchmark result

1. finite-sample adequacy calibration on a grid of `(n, L, H, sigma0)`
2. FWLS-oracle equivalence diagnostics
3. empirical rate-constant stabilization for `RMSE(\hat H)`
4. power transition near the minimax boundary

### Open problem

1. robustness under misspecified scale laws
2. robust sandwich-style calibration under weight misspecification
3. high-dimensional extension via sliced or entropic transport
4. sequential or anytime-valid testing
5. controller switching under local stationarity after rejection

V1 should not claim theorem closure for the open-problem items.

## V1 evidence package

Main figures:

1. null calibration
2. minimax-boundary power
3. FWLS-oracle agreement
4. rate-constant stabilization
5. robustness to scale misspecification and alternative noise laws

Main tables:

1. null calibration grid
2. minimax-boundary power grid
3. rate-constant grid
4. misspecification sensitivity table
5. noise-robustness table

The main text should keep the five figures in the narrative and push grid
details to the appendix.

## Python package for V1

Create a dedicated package rather than mixing the spin-off with the main-paper
modules.

Proposed package:

`code/scale_consistency/`

Proposed modules:

1. `model.py`
   - generate exact `D_j = zeta j^H`
   - generate noisy observations `\hat D_j`
   - generate alternatives indexed by `kappa`
   - generate misspecified scale profiles for robustness checks

2. `estimation.py`
   - pilot OLS
   - oracle WLS
   - feasible WLS
   - residual statistic `Q`
   - convenience wrappers returning `\hat H`, `\hat alpha`, fitted weights, residuals

3. `theory_diagnostics.py`
   - asymptotic variance formulas
   - boundary rate `kappa*(n, L, H)`
   - finite-sample summaries used in tables

4. `experiments.py`
   - null-calibration experiment
   - FWLS-versus-oracle experiment
   - minimax-boundary power experiment
   - rate-constant experiment
   - misspecification experiment
   - noise-robustness experiment

5. `plots.py`
   - figure builders for all V1 figures

6. `report.py`
   - write CSV summaries
   - write LaTeX tables
   - save figures into `artifacts/`

Proposed test file set:

1. `tests/test_scale_consistency_model.py`
2. `tests/test_scale_consistency_estimation.py`
3. `tests/test_scale_consistency_theory_diagnostics.py`
4. `tests/test_scale_consistency_experiments.py`

The unit tests should verify identities, invariances, and monotonicities. They
should not rely on fragile Monte Carlo thresholds except where unavoidable.

## Experiment matrix

### E1. Null calibration

Goal: support the null-law section.

Grid:

1. `L in {10, 20, 30, 50}`
2. `n in {200, 500, 1000, 2000, 5000}`
3. `H in {0.3, 0.6, 0.8}`
4. `sigma0 in {0.5, 1.0, 2.0}`

Outputs:

1. empirical rejection rate at nominal `5%`
2. mean and variance of `Q`
3. KS or Cramer-von Mises discrepancy against `chi^2(L-2)`

### E2. FWLS versus oracle

Goal: support the feasible-transfer theorem empirically.

Outputs:

1. `RMSE(\hat H_fwls - \hat H_oracle)` versus `n`
2. variance ratio `Var(\hat H_fwls) / Var(\hat H_oracle)`
3. difference in test statistics `|Q_fwls - Q_oracle|`

### E3. Boundary power

Goal: support the minimax section.

Grid:

1. `c in {0.25, 0.5, 1.0, 1.5, 2.0, 3.0}`
2. `kappa = c * (nL)^(-1/2)` in the fixed-`L` theorem regime

Outputs:

1. power curve versus `c`
2. power surface over `(n, c)` for one representative `L`

### E4. Rate constant

Goal: support the estimation section.

Outputs:

1. `RMSE(\hat H)` versus `n`
2. scaled constant `C = RMSE(\hat H) * (n L^{2H+1})^(1/2)`
3. oracle counterpart for comparison

### E5. Scale misspecification

Goal: evidence for the discussion section, not theorem closure.

Model family:

`D_j = zeta j^H (1 + b g_j)` with small perturbation amplitude `b`.

Perturbation examples:

1. local bump in `log j`
2. sinusoidal modulation in `log j`
3. piecewise slope shift

Outputs:

1. size inflation under `b > 0`
2. sensitivity of `\hat H`
3. instability of the `chi^2` calibration

### E6. Noise misspecification

Goal: evidence for scope limits.

Noise families:

1. Gaussian benchmark
2. subgaussian bounded noise
3. standardized `t` noise with moderate degrees of freedom

Outputs:

1. empirical size
2. empirical power at the boundary

## Figure package

V1 should carry only five main figures.

1. `fig_null_calibration.pdf`
   - QQ or CDF comparison between empirical `Q` and `chi^2(L-2)`

2. `fig_power_boundary.pdf`
   - power versus `c = kappa / kappa*`

3. `fig_fwls_oracle_gap.pdf`
   - estimator/test-statistic gap versus `n`

4. `fig_rate_constant.pdf`
   - RMSE and scaled constant stabilization

5. `fig_misspecification_sensitivity.pdf`
   - size or calibration drift under scale misspecification

Optional sixth figure only if compact and informative:

6. `fig_noise_robustness.pdf`

## Table package

V1 should carry only four main tables.

1. `tab_null_size.tex`
2. `tab_boundary_power.tex`
3. `tab_rate_constant.tex`
4. `tab_misspecification_summary.tex`

Extended grids belong in repository CSVs, not in the main text.

## Artifact layout

Proposed outputs:

1. `artifacts/csv/scale_consistency/null_calibration.csv`
2. `artifacts/csv/scale_consistency/fwls_oracle.csv`
3. `artifacts/csv/scale_consistency/boundary_power.csv`
4. `artifacts/csv/scale_consistency/rate_constant.csv`
5. `artifacts/csv/scale_consistency/misspecification.csv`
6. `artifacts/csv/scale_consistency/noise_robustness.csv`
7. `artifacts/figures/scale_consistency/*.pdf`
8. `artifacts/tables/scale_consistency/*.tex`

## Execution order

1. Implement `model.py` and `estimation.py`.
2. Add unit tests for identities and oracle/feasible consistency checks.
3. Implement E1 and E2 first; these are closest to the theorem line.
4. Implement E3 and E4 next; these complete the core empirical package.
5. Implement E5 and E6 last; these feed the discussion and limitations.
6. Only after figures and tables are stable, rewrite the numerical section in
   final paper prose.

## Acceptance criteria for V1

The spin-off is V1-ready when:

1. all theorem statements are stable in the manuscript
2. all four main tables are generated from code
3. all five main figures are generated from code
4. the core Python tests pass
5. each figure and table has a single source script and a single CSV input
6. misspecification and robustness results are explicitly labeled as evidence,
   not as theorem closure
