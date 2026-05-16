# Main Theorem Package

## Core law

The main theorem package revolves around the carrier-roughness useful-memory horizon law rather than any single carrier instantiation.

Its primary object is the temporal validity of retained evidence for distribution tracking under drift. The package is adjacent to dynamic regret and adaptive windowing, but it is organized around a validity horizon, not around regret against a moving comparator or the stopping time of a reaction rule.

The governing upper envelope is:

`error_t(n) <= C_K n^{-a} + C_S zeta n^H`

where:

- `a > 0` is the finite-sample carrier exponent for the measurement layer;
- `H in (0,1]` is the temporal roughness exponent for the drift path;
- `zeta` is the drift amplitude;
- `C_K` and `C_S` are regime-dependent constants.

## Theorem 1: Abstract upper law

Assume:

- a carrier bound `E d(\hat P_t^{(n)}, \bar P_t^{(n)}) <= C_K n^{-a}` for the empirical/windowed object relative to the within-window target `\bar P_t^{(n)}`;
- a staleness bound `d(\bar P_t^{(n)}, P_t) <= C_S zeta n^H` for the drift path class.

Then

`E d(\hat P_t^{(n)}, P_t) <= C_K n^{-a} + C_S zeta n^H`.

This is the theorem-level object of the paper.

In particular, the horizon is the statistical object being characterized, not just a tuning parameter for a detector or an online-learning policy.

Status:

- theorem-ready once the carrier and staleness assumptions are stated for the chosen measurement layer and path class;
- the proof is immediate from the triangle inequality plus expectation;
- the main remaining work is not this theorem itself but supplying rigorous carrier instantiations.

## Corollary 1: Optimized useful-memory horizon law

Balancing the two terms yields the useful-memory horizon

`n^*(a,H) ~ (C_K / zeta)^{1/(a+H)}`

up to constant factors, and the corresponding optimized error scale

`R^*(a,H,zeta) ~ C_K^{H/(a+H)} zeta^{a/(a+H)}`

again up to constant factors.

Special cases:

- `a = 1/2`, `H = 1` gives the cube-root horizon `n^* ~ (C_K / zeta)^{2/3}`;
- `a = 1/2`, general `H` gives the corresponding `H`-indexed subfamily `n^* ~ (C_K / zeta)^{2/(1+2H)}`.

## Theorem 2: Structural lower bound

The lower-bound half of the package shows that the optimized scale is structural, not an estimator artifact.

The lower-bound target is a roughness-matched Gaussian witness subclass:

`mu_{t-j}^{\pm,h} = \pm beta (h-j)_+^H`

with a Le Cam / Pinsker argument giving a matching exponent at least in the `a = 1/2` regime.

The key scientific role of the lower bound is to certify that the useful-memory horizon law is not just the optimum of one procedure.

Status:

- theorem-ready at the structural level for the main `a = 1/2` slice;
- the consolidated note is `notes/structural_lower_bound.md`;
- class-tight and constant-sharp extensions remain open.

## Proposition 3: Minimum kernel instantiation

In the bounded-support fixed-span minimum kernel, the carrier satisfies

`E W_2(\hat P_n^{tri}, \bar P_n) = O(n^{-1/2})`

under the 1-D quantile/Bahadur route with interior regularity.

This provides the canonical `a = 1/2` instantiation of the abstract upper law.

Status:

- theorem-ready in the safe-zone assumptions collected in `notes/minimum_kernel_proposition.md`;
- the natural first statement is `E W_2^2(\hat P_n^{tri}, \bar P_n) = O(n^{-1})`, followed by Jensen;
- the consolidated proof narrative now lives in `notes/minimum_kernel_proof.md`.

## Proposition 4: Useful-layer instantiation

In low-dimensional / low-intrinsic-dimensional fixed-span regimes, the triangular array should inherit the i.i.d. mixture benchmark carrier exponent `a` up to a small constant-level gap.

The useful-layer bridge is:

- bounded support;
- fixed span;
- embedded low-intrinsic support;
- triangular and i.i.d. mixture slopes remain close.

This is the bridge from the minimum kernel to the useful layer.

Status:

- supported by the current low-intrinsic-dimension experiments;
- theorem-target note is `notes/useful_layer_bridge.md`;
- not yet a closed theorem in the same sense as the minimum kernel target.

## Proposition 5: Practically relevant instantiation

For a dimension-robust measurement layer such as fixed-`epsilon` debiased Sinkhorn, the goal is to obtain a stable carrier exponent that can feed directly into the abstract law.

This is not yet the main closed theorem. It is the next measurement-layer instantiation once the useful bridge is in place.

Status:

- empirical signal exists for fixed-`epsilon` Sinkhorn;
- theorem-target note is `notes/practical_layer_measurement.md`;
- no claim should currently say that the practical layer is closed as a theorem.

## What is already enough for the paper

The package has a real contribution if it delivers:

- the abstract upper law;
- the optimized horizon corollary;
- a structural lower bound with matching exponent at the main slice;
- at least one rigorous carrier instantiation;
- supporting evidence for the useful and practically relevant layers.

This is already a complete theory in the relevant sense: the law is abstract, the carriers are modular, and the lower bound makes the horizon structural.

## Current status line

What is already closed or theorem-ready:

- the abstract upper law;
- the horizon optimization corollary;
- the minimum-kernel proposition in theorem-ready form;
- the structural role of the lower bound at the main `a = 1/2` slice.

What is still open at paper level:

- porting the consolidated minimum-kernel proof into final manuscript form;
- the useful-layer inheritance theorem beyond experimental evidence;
- the practically relevant measurement-layer theorem;
- lower-bound extensions beyond the main slice.

The boundary between theorem-ready results, theorem targets, conjectures, and future work is tracked in `notes/package_boundaries.md`.

## What should stop consuming time

The paper does not need every carrier instantiation to be fully closed before it becomes real.

In particular:

- the minimum kernel should not absorb unlimited effort once it is theorem-ready;
- the useful layer only needs to be precise enough to serve as the first bridge theorem;
- the practically relevant layer can remain a theorem target plus strong empirical support if the abstract law and lower bound are solid.
