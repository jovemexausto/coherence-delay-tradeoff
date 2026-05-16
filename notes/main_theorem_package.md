# Main Theorem Package

## Core law

The main theorem package revolves around the carrier-roughness useful-memory horizon law rather than any single carrier instantiation.

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

## Proposition 3: Minimum kernel instantiation

In the bounded-support fixed-span minimum kernel, the carrier satisfies

`E W_2(\hat P_n^{tri}, \bar P_n) = O(n^{-1/2})`

under the 1-D quantile/Bahadur route with interior regularity.

This provides the canonical `a = 1/2` instantiation of the abstract upper law.

## Proposition 4: Useful-layer instantiation

In low-dimensional / low-intrinsic-dimensional fixed-span regimes, the triangular array should inherit the i.i.d. mixture benchmark carrier exponent `a` up to a small constant-level gap.

The useful-layer bridge is:

- bounded support;
- fixed span;
- embedded low-intrinsic support;
- triangular and i.i.d. mixture slopes remain close.

This is the bridge from the minimum kernel to the useful layer.

## Proposition 5: Practically relevant instantiation

For a dimension-robust measurement layer such as fixed-`epsilon` debiased Sinkhorn, the goal is to obtain a stable carrier exponent that can feed directly into the abstract law.

This is not yet the main closed theorem. It is the next measurement-layer instantiation once the useful bridge is in place.

## What is already enough for the paper

The package has a real contribution if it delivers:

- the abstract upper law;
- the optimized horizon corollary;
- a structural lower bound with matching exponent at the main slice;
- at least one rigorous carrier instantiation;
- supporting evidence for the useful and practically relevant layers.

This is already a complete theory in the relevant sense: the law is abstract, the carriers are modular, and the lower bound makes the horizon structural.

## What should stop consuming time

The paper does not need every carrier instantiation to be fully closed before it becomes real.

In particular:

- the minimum kernel should not absorb unlimited effort once it is theorem-ready;
- the useful layer only needs to be precise enough to serve as the first bridge theorem;
- the practically relevant layer can remain a theorem target plus strong empirical support if the abstract law and lower bound are solid.
