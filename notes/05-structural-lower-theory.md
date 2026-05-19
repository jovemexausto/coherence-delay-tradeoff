# 05. Structural Lower Theory
Status: closed
Category: lower
Prev: 04. One-Dimensional Proof Details
Next: 06. Exact Gaussian Witness Frontier

Exponent-level lower theory supporting the main manuscript.

## Goal

Show that, even on a restricted Gaussian location subclass, no estimator can
beat the horizon scale associated with the root-`n` finite-sample regime.

## Construction

Use the Gaussian path pair

- `mu_{t-j}^{\pm,h} = \pm beta (h-j)_+^H`

with `beta <= zeta`.

## Lower law

The KL divergence across the retained window is of order

- `beta^2 sum_{r=1}^h r^{2H} / sigma^2`.

Balancing the roughness budget with the testing budget yields

- horizon scale `h^*(H) ~ (sigma/zeta)^{2/(2H+1)}`;
- lower law `Risk >= c_H sigma^{2H/(2H+1)} zeta^{1/(2H+1)}`.

## Status

This is a structural subclass lower bound. It is enough for the central
claim that the horizon is not an estimator artifact. It is not a class-tight
distributional lower theorem.

## Supporting refinement

The exact Gaussian two-point calculation and the endpoint-minimal profile
refinement sharpen the constants on this same exponent scale. Those details are
secondary to the main lower-law role and now live in the paper appendix.
