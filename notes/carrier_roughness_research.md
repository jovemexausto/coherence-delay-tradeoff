# Carrier-Roughness Research Note

## Formal conjectures

1. Carrier inheritance at fixed span.
For a windowed triangular array with independent samples `X_{t-j} ~ P^*_{t-j}` and fixed within-window span `zeta n^H <= S`, the windowed empirical law should inherit the same carrier exponent `a` as the i.i.d. benchmark from the mixture `\bar P_t^{(n)}`.

2. Carrier identification is geometric.
For raw `W_2`, the carrier exponent should be controlled by effective Wasserstein dimension: `a = 1/2` in low-dimensional or low-intrinsic-dimensional regimes, and `a < 1/2` once the volumetric barrier dominates.

3. Sinkhorn restores the statistical carrier.
For fixed `epsilon > 0`, the debiased Sinkhorn measurement layer should preserve a root-`n` carrier exponent `a = 1/2` across dimensions, with the dependence moved into constants rather than exponents.

4. Joint carrier-roughness horizon law.
If the finite-sample term scales as `C_K n^{-a}` and staleness grows as `zeta n^H`, then the useful-memory horizon should satisfy `n^*(a,H) ~ (C_K / zeta)^{1 / (a + H)}`.

5. Lower-bound frontier.
The Lipschitz Gaussian witness is exponent-tight at `H = 1`. Matching lower bounds for `H in (0,1)` should require roughness-matched witness paths rather than only the linear ramp geometry.

6. Holder witness conjecture.
For the Gaussian location subclass with witness path `mu_{t-j}^{\pm,h} = \pm beta (h-j)_+^H`, the same Le Cam / Pinsker argument should yield a lower law of order `sigma^{2H/(2H+1)} zeta^{1/(2H+1)}`. This would match the `(a,H)` family when `a = 1/2`, still at a subclass-based rather than class-tight level.

## Priority numerical tests

1. I.i.d. mixture benchmark stability.
Check whether the benchmark slope stays stable as the window size changes under fixed-span scaling.

2. Triangular-array inheritance.
Compare `iid-mixture` and `triangular` slopes under the same fixed-span schedule. The main empirical target is whether they match within noise.

3. Heterogeneity stress test.
Repeat the same comparison under fixed `zeta`, where span grows with `n`. This is the candidate regime where inheritance may visibly break.

4. Ambient versus intrinsic dimension.
Contrast raw `W_2` slopes for ambient cubes with embedded low-dimensional supports inside higher-dimensional spaces.

5. Sinkhorn epsilon sweep.
Track how the estimated carrier changes with `epsilon` for both `iid-mixture` and `triangular` comparisons. The target signature is exponent stability with shifting constants.

6. Holder witness sweep.
For `H in (0,1]`, optimize the roughness-matched Gaussian witness numerically and compare the normalized optimum against the asymptotic constant predicted by the Holder ramp calculation.

## Theorem targets

### Conservative

Assume a carrier law `E W_2(\hat P_t^{(n)}, \bar P_t^{(n)}) <= C_K n^{-a}` and derive the full `(a,H)` carrier-roughness horizon family. Keep the triangular-array carrier identification and the general-H lower matching as explicit future work.

### Moderate

Prove triangular-array inheritance of the i.i.d. mixture carrier under strong conditions: bounded support, uniform moment control, low effective dimension, and fixed within-window span. This would justify the root-`n` slice rigorously in the most relevant low-dimensional regime.

### Ambitious

Establish a measurement-layer theorem for fixed-`epsilon` debiased Sinkhorn and combine it with a roughness-indexed drift law to obtain a fully rigorous `(a,H)` horizon theory with `a = 1/2` at the measurement layer. Parallel to that, extend the lower-bound program from the Lipschitz endpoint to roughness-matched Holder witnesses.

## New computational support

- `umh-research-carrier-roughness` runs empirical carrier identification sweeps for raw `W_2`, intrinsic-dimension proxies, triangular-array inheritance, and fixed-`epsilon` Sinkhorn.
- `umh-research-holder-lower-bound` runs the roughness-matched Gaussian witness sweep for `H in (0,1]`, reporting the optimal witness width and the normalized lower-law constant.
