# 13. Sinkhorn Self-Coupling Lemma
Status: active
Category: theorem target
Prev: 12. Theorem Ladder
Next: -

## Target statement

The bounded-cardinality finite-discrete embedded theorem is closed. The active
target is the calibrated support-growth family on a moderate band
`B=[eps_min, eps_max]` bounded away from zero.

The relevant linearized self-coupling operator is the squared centered block
`S_eps^2`, not `S_eps`. The active route is therefore:

1. keep the centered self-coupling contraction explicit via the same minorization route;
2. identify a covariance-weighted quadratic Sinkhorn null bound in `\ell_2` on the support-growth family;
3. combine that bound with the exact calibrated block-isotropic fluctuation identity;
4. extend the mechanism from support growth to support change.

The calibrated numerical evidence is now favorable on `(8,2)` and `(12,2)`.
On the discrete calibrated `k=2` support grid with span `0.25` and
`B=[0.1,0.5]`, the worst centered squared radius is `0.116`, the worst inverse
norm is `1.132`, the exact-target null slopes lie in `[-1.001,-0.878]` for
`(8,2)` and `[-0.878,-0.778]` for `(12,2)`, and the root-`n` finite-support
fluctuation proxy remains bounded on the tested sample range.

## Proof route

1. Rewrite the coupled Sinkhorn equations in centered coordinates.
2. Eliminate one coordinate block and identify the reduced operator as `S_eps^2`.
3. Use positivity/reversibility on the compact moderate band to obtain the same
   Doeblin gap for `S_eps^2|V_0` on the support-growth family.
4. Express the null comparison directly in terms of the empirical weight error.
5. Prove a covariance-weighted quadratic bound compatible with the block-isotropic
   empirical fluctuations.
6. Insert the exact calibrated identity `E||\Delta b||_2^2 = 3/(4n)` and the
   block-basis covariance formula.

On the calibrated discrete `k=2` support grid, the compactness step can be made
fully explicit. The support lies in a rectangle of squared diameter
`D^2 = 5/16`, so for every `epsilon in B` the kernel entries satisfy
`K_ij >= exp(-D^2/epsilon_min)`. After normalization, the centered transition
kernel decomposes as a convex combination of the uniform kernel and a stochastic
remainder, which gives a uniform contraction on the centered subspace. This is
the concrete minorization route behind the `S_eps^2` gap bound.

## Proof status

- the centered gap is controlled by finite-state minorization on the calibrated
  grid and on bounded finite-support classes;
- the exact calibrated support-growth fluctuation identity is closed:
  `E||\Delta b||_2^2 = 3/(4n)`, with block-basis covariance `(4n^2)^{-1}I`;
- numerical Hessian probes show bounded local curvature but collective curvature
  growing linearly in `n`, so a uniform operator bound looks too strong;
- the missing support-growth step is a covariance-weighted quadratic null bound.
