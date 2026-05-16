# Structural Lower Bound

This note states the lower-bound half of the theorem package in a theorem-ready form for the main slice `a = 1/2`.
Its role is to show that the useful-memory horizon is structural, not an artifact of a particular estimator.

## Goal

Show that even on a restricted Gaussian subclass, no estimator can beat the main useful-memory scale once the carrier slice is `a = 1/2`.

For the main slice, the target lower law is of order

`sigma^{2H/(2H+1)} zeta^{1/(2H+1)}`

with the corresponding horizon scale

`h^*(H) ~ (sigma / zeta)^{2/(2H+1)}`.

At `H = 1`, this recovers the cube-root horizon.

## Witness subclass

Work in the Gaussian location model with known variance `sigma^2` and path means

`mu_{t-j}^{\pm,h} = \pm beta (h-j)_+^H`

for `j = 0, 1, ..., h`, where `beta <= zeta` enforces the roughness budget.

This is a subclass construction, not a full minimax characterization of all deterministic Hölder path classes.

## Two-point reduction

Compare two hypotheses:

- `P_+`: the window is generated from the `+` witness path;
- `P_-`: the window is generated from the `-` witness path.

At the present time, the separation of the targets is of order

`W_2(P_{+,t}, P_{-,t}) ~ beta h^H`.

In the Gaussian location subclass, `W_2` between equal-variance Gaussians is exactly the mean gap, so the signal size is controlled directly by the endpoint separation.

## KL budget

For the whole window, the KL divergence between the two hypotheses is proportional to

`beta^2 sum_{r=1}^h r^{2H} / sigma^2`.

The standard Le Cam choice is to set `beta` as large as the roughness budget allows while keeping the KL divergence bounded by a small constant. This gives

`beta = min(zeta, sigma / (2 sqrt(sum_{r=1}^h r^{2H})))`.

That is the exact choice implemented in `code/useful_memory_horizon/Hölder_lower_bound_research.py`.

## Lower-bound template

With the KL budget controlled, a two-point testing inequality yields a risk lower bound of order

`beta h^H`.

Balancing the two regimes for `beta` yields the critical horizon scale

`h^*(H) ~ (sigma / zeta)^{2/(2H+1)}`

and the corresponding lower law

`Risk >= c_H sigma^{2H/(2H+1)} zeta^{1/(2H+1)}`

for a constant `c_H > 0` depending only on `H` within this witness construction.

## Main slice: `H = 1`

At `H = 1`,

- `h^* ~ (sigma / zeta)^{2/3}`;
- `Risk >= c sigma^{2/3} zeta^{1/3}`.

This matches the upper-law optimization when the carrier slice is `a = 1/2` and staleness is linear.

So the cube-root horizon is not just the optimum of one procedure. It already appears as a structural lower-bound scale on the witness subclass.

## General `H in (0,1]`

The same witness calculation predicts

- horizon scale `h^*(H) ~ (sigma / zeta)^{2/(2H+1)}`;
- lower law `sigma^{2H/(2H+1)} zeta^{1/(2H+1)}`.

This matches the abstract `(a,H)` law on the main carrier slice `a = 1/2`.

Current status:

- theorem-ready as a subclass lower bound for the main slice logic;
- strongest closed interpretation remains the structural role of the lower bound;
- full class-tight minimax statements and constant-sharp extensions remain open.

## What is already supported

The numerical witness sweep in `umh-research-Hölder-lower-bound` supports this scaling by:

- optimizing the witness width `h`;
- checking the normalized lower-bound value against the predicted asymptotic constant;
- confirming the `sigma` and `zeta` exponents across `H in (0,1]`.

## What this note proves and what it does not

This note is enough for the paper's structural message because it shows that the main useful-memory scale has a lower-bound witness.

This note does not claim:

- a class-tight minimax theorem for all deterministic Hölder paths;
- sharp constants for the full path class;
- a lower bound for carrier slices other than `a = 1/2`.

Those belong to later extensions, not to the minimal theorem package.
