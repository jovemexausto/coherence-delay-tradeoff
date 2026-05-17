# 06. Exact Gaussian Witness Frontier
Status: active
Category: refinement
Prev: 05. Structural Lower Theory
Next: 07. Witness Shape Extremality

Use the exact Gaussian two-point testing error on the witness family

`mu_{t-j}^{\pm,h} = \pm beta (h-j)_+^H`.

This does not replace the main structural lower bound. It identifies the next
constant-level target for that lower regime.

It also treats the ramp witness itself. The separate note
`notes/07-witness-shape-extremality.md` studies whether that ramp is shape-optimal
inside the larger endpoint-saturating Hölder witness class.

## Fixed `h`: exact two-point reduction

For the Gaussian location witness with common variance `sigma^2`, the whole-window
mean difference has squared norm

`4 beta^2 S_h`, where `S_h = sum_{r=1}^h r^{2H}`.

The exact Bayes testing error for the two hypotheses is therefore

`Phi(- beta sqrt(S_h) / sigma)`.

So the two-point endpoint-risk lower bound on this witness family becomes

`L_h(beta) = beta h^H Phi(- beta sqrt(S_h) / sigma)`.

For fixed `h`, the unconstrained optimizer solves

`Phi(-x_0) = x_0 phi(x_0)`,

with numerical root

`x_0 = 0.7517915246935645...`.

Hence the exact fixed-`h` optimizer is

`beta_h^* = min(zeta, x_0 sigma / sqrt(S_h))`.

## Large-ratio asymptotics after optimizing `h`

In the roughness-active regime, `beta = zeta` and

`S_h ~ h^{2H+1} / (2H+1)`.

Write

`p_H = 2H / (2H + 1)`

and

`x = zeta sqrt(S_h) / sigma ~ zeta h^{H+1/2} / (sigma sqrt(2H+1))`.

After normalizing by `sigma^{p_H} zeta^{1-p_H}`, the objective reduces to

`F_H(x) = (2H+1)^{H/(2H+1)} x^{p_H} Phi(-x)`.

The optimizer `x_H` therefore solves the scalar equation

`p_H Phi(-x_H) = x_H phi(x_H)`.

This yields the exact-Gaussian asymptotic candidate constant

`C_H^{Gauss} = (2H+1)^{H/(2H+1)} x_H^{p_H} Phi(-x_H)`

and the corresponding horizon shape parameter

`A_H^{Gauss} = (sqrt(2H+1) x_H)^{2/(2H+1)}`,

so that

`h_H^* ~ A_H^{Gauss} (sigma / zeta)^{2/(2H+1)}`.

## Numerical status

The module `code/useful_memory_horizon/gaussian_witness_frontier.py` verifies this
reduction against direct discrete optimization over `h`.

For `sigma / zeta in {10^3, 10^4}` and `H in {0.35, 0.5, 0.75, 1.0}`, the
normalized optimum matches `C_H^{Gauss}` to within about `5e-3` relative error,
and the normalized optimal horizon matches `A_H^{Gauss}` equally closely.

The exported summary table is `artifacts/csv/gaussian_witness_frontier/summary.csv`.

## Comparison with the current Pinsker-style witness constant

The exact Gaussian testing bound strictly improves the current Pinsker-based
constant from `holder_lower_bound_research.py` on the same witness family.

At `sigma / zeta = 10^4`, the numerical improvement factors are approximately:

- `1.108` at `H = 0.35`;
- `1.138` at `H = 0.5`;
- `1.174` at `H = 0.75`;
- `1.199` at `H = 1.0`.

So this does not change the lower-bound exponent or the structural horizon law,
but it does sharpen the lower-regime constant in a way that now has a clean
one-dimensional variational characterization.

This still treats the ramp witness. The companion note
`notes/07-witness-shape-extremality.md` shows that for `H < 1` the ramp is not the
lowest-energy endpoint-saturating Hölder profile, so the exact Gaussian ramp
frontier is itself not the final lower-shape frontier.

## Status discipline

This note sharpens the lower-regime constant on a specific Gaussian witness
family. It should not be inflated into:

- a class-tight lower theorem for all Hölder path classes;
- a complete lower theory for arbitrary carrier exponent `a`;
- or a completed sharp theory for the whole `(a,H)` family.
