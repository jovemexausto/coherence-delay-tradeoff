# Carrier-Roughness Research Note

This note supports the abstract carrier side of `notes/main_theorem_package.md`.
Its role is to identify which carrier instantiations are strong enough to feed the carrier-roughness useful-memory horizon law.

## Formal conjectures

1. Carrier inheritance at fixed span.
For a windowed triangular array with independent samples `X_{t-j} ~ P^*_{t-j}` and fixed within-window span `zeta n^H <= S`, the windowed empirical law should inherit the same carrier exponent `a` as the i.i.d. benchmark from the mixture `\bar P_t^{(n)}`.

2. Carrier identification is geometric.
For raw `W_2`, the carrier exponent should be controlled by effective Wasserstein dimension: `a = 1/2` in low-dimensional or low-intrinsic-dimensional regimes, and `a < 1/2` once the volumetric barrier dominates.

3. Sinkhorn may stabilize the measurement-layer carrier.
For fixed `epsilon > 0`, the debiased Sinkhorn measurement layer is conjectured to preserve a stable carrier exponent across dimensions, with more of the dimensional dependence moved into constants than in raw `W_2`.

4. Joint carrier-roughness useful-memory horizon law.
If the finite-sample term scales as `C_K n^{-a}` and staleness grows as `zeta n^H`, then the useful-memory horizon should satisfy `n^*(a,H) ~ (C_K / zeta)^{1 / (a + H)}`.

5. Lower-bound frontier.
The Lipschitz Gaussian witness is exponent-tight at `H = 1`. Matching lower bounds for `H in (0,1)` should require roughness-matched witness paths rather than only the linear ramp geometry.

6. Hölder witness conjecture.
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

6. Hölder witness sweep.
For `H in (0,1]`, optimize the roughness-matched Gaussian witness numerically and compare the normalized optimum against the asymptotic constant predicted by the Hölder ramp calculation.

## Theorem targets

### Conservative

Assume a carrier law `E W_2(\hat P_t^{(n)}, \bar P_t^{(n)}) <= C_K n^{-a}` and derive the full `(a,H)` useful-memory horizon family. Keep the triangular-array carrier identification and the general-H lower matching as explicit future work.

### Moderate

Prove triangular-array inheritance of the i.i.d. mixture carrier under strong conditions: bounded support, uniform moment control, low effective dimension, and fixed within-window span. This would justify the `a=1/2` slice rigorously in the most relevant low-dimensional regime.

### Current moderate theorem target

Under bounded support, fixed span, and low effective dimension, the triangular-array carrier should match the i.i.d. mixture carrier up to a small constant-level gap. The point is not that the raw `W_2` rate becomes dimension-free, but that the triangular window inherits the same exponent as the i.i.d. mixture benchmark in the regime where effective dimension is low enough for the carrier to remain near `1/2`.

Operationally, the numerical target is:

- `a_tri` and `a_iid` both stay near the `a=1/2` carrier slice;
- `|a_tri - a_iid|` stays small under fixed span;
- increasing ambient dimension without increasing intrinsic dimension does not destroy the carrier.

That is the bridge from Proposition 3 to Proposition 4 in the main theorem package.

### Moderate theorem formulation

The cleanest current statement is:

- on a bounded-support window with fixed span;
- for an embedded low-intrinsic-dimension support;
- the triangular-array `W_2` carrier exponent matches the i.i.d. mixture benchmark up to a small constant-level gap.

In the present lab, the strongest stable signal is for an embedded `k=1` support inside a larger ambient space. With the same fixed-span schedule, both the triangular window and the i.i.d. mixture benchmark remain near the `a=1/2` carrier slice, and the slope gap stays small.

This is the first theorem-shaped bridge from the minimum kernel to the useful layer.

The practical numerical formulation now used by the lab is:

- `a_tri > 0.45` and `a_iid > 0.45` in the `k=1` embedded case;
- `|a_tri - a_iid| < 0.08` under fixed span;
- increasing ambient dimension while keeping intrinsic dimension fixed does not destroy the carrier.

That is the moderate theorem in the form the current experiments can support.

### Ambitious

Establish a measurement-layer theorem for fixed-`epsilon` debiased Sinkhorn and combine it with a roughness-indexed drift law to obtain a fully rigorous `(a,H)` horizon theory with `a = 1/2` at the measurement layer. Parallel to that, extend the lower-bound program from the Lipschitz endpoint to roughness-matched Hölder witnesses.

## New computational support

- `umh-research-carrier-roughness` runs empirical carrier identification sweeps for raw `W_2`, intrinsic-dimension proxies, triangular-array inheritance, and fixed-`epsilon` Sinkhorn.
- `umh-research-Hölder-lower-bound` runs the roughness-matched Gaussian witness sweep for `H in (0,1]`, reporting the optimal witness width and the normalized lower-law constant.

## Current lab signal for the next layer up

Two focused sweeps now support the `Useful` and `Practically relevant` layers in the theorem ladder.

### Low intrinsic dimension, bounded support

For an embedded cube supported on a 1-D or 2-D intrinsic subspace inside a higher ambient space, the raw `W_2` slopes move back toward the `a=1/2` carrier slice.

Representative sweep on `ambient_dim = 8`:

- intrinsic `k = 1` gives carrier `a \approx 0.47-0.50`;
- intrinsic `k = 2` gives carrier `a \approx 0.45`;
- the same support in a full `d = 8` ambient cube is slower.

This is the strongest direct evidence so far for the `Useful` layer: the carrier is controlled by effective dimension, not ambient dimension alone.

A more robust sweep with larger `n` makes the same point cleaner:

- `ambient_dim = 4`, `intrinsic_dim = 1` gives triangular raw-`W_2` carrier `a \approx 0.46-0.48`;
- `ambient_dim = 8`, `intrinsic_dim = 1` gives `a \approx 0.47-0.48`;
- `ambient_dim = 8`, `intrinsic_dim = 2` gives `a \approx 0.42-0.43`.

The key qualitative fact is that the `k=1` embedded support behaves similarly across ambient dimensions, while increasing intrinsic dimension slows the carrier.

### Fixed-`epsilon` Sinkhorn on the same low-intrinsic geometry

On the same embedded low-intrinsic support, the debiased Sinkhorn proxy produces a stable exponent across `\epsilon` values.

Representative sweep in `ambient_dim = 8`, `intrinsic_dim = 1`:

- `\epsilon = 0.50` -> triangular carrier `a \approx 0.70`, i.i.d. benchmark `a \approx 0.74`
- `\epsilon = 0.20` -> triangular carrier `a \approx 0.69`, i.i.d. benchmark `a \approx 0.72`
- `\epsilon = 0.10` -> triangular carrier `a \approx 0.68`, i.i.d. benchmark `a \approx 0.71`
- `\epsilon = 0.05` -> triangular carrier `a \approx 0.67`, i.i.d. benchmark `a \approx 0.70`

The exact exponent proxy is still measurement-layer specific, but the important signature is that the triangular and i.i.d. slopes remain close while `\epsilon` primarily moves constants and mildly shifts the effective exponent. That is the right qualitative sign for the practical layer.

A larger sweep on `ambient_dim = 8`, `intrinsic_dim = 1` gives:

- `\epsilon = 0.50` -> triangular `a \approx 0.460`, i.i.d. `a \approx 0.527`
- `\epsilon = 0.20` -> triangular `a \approx 0.459`, i.i.d. `a \approx 0.531`
- `\epsilon = 0.10` -> triangular `a \approx 0.451`, i.i.d. `a \approx 0.534`
- `\epsilon = 0.05` -> triangular `a \approx 0.449`, i.i.d. `a \approx 0.534`

So the practical-layer signal is not yet a closed theorem, but it is consistent with a stable measurement-layer carrier and with `\epsilon` behaving as a constant-level control knob rather than a qualitative phase change.
