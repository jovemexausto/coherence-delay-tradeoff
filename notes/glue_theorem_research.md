# Glue Theorem Research Note

This note is now best read as background for the minimum kernel and its fixed-span design effect.
It supports Proposition 3 in `notes/main_theorem_package.md`, rather than defining the main paper contribution.

## Current P2 target

The current goal is a theorem in the 1-D Gaussian location model for the windowed triangular array under fixed within-window span.

Given independent samples `X_j ~ N(mu_j, sigma^2)` and the window mixture

`bar P_n = (1/n) sum_j N(mu_j, sigma^2)`,

the target is to justify

`E W_2(hat P_n^{tri}, bar P_n) <= C(Delta, sigma) n^{-1/2}`

with `Delta = max_j mu_j - min_j mu_j` bounded as `n` grows.

## What the numerics now support

1. Fixed-span inheritance.
The triangular array and the i.i.d. mixture benchmark show nearly identical `n^{-1/2}` slopes in the 1-D Gaussian fixed-span experiments.

2. A curvature constant `kappa(Delta)`.
The quantile-process numerics suggest a constant of the form

`kappa(Delta)^2 = integral A_tri(u) du`,

where `A_tri(u)` is the triangular-array quantile variance integrand.

3. Homogeneous-window baseline.
At `Delta = 0`, the asymptotic constant reduces to a one-dimensional Gaussian integral that we evaluate numerically. This gives the homogeneous benchmark `kappa(0)` used as the fixed-span reference level.

4. Concavity improvement over the i.i.d. mixture benchmark.
In the 1-D Gaussian location model, the triangular-array asymptotic integrand is pointwise below the i.i.d. benchmark integrand because `x(1-x)` is concave. This suggests `C_tri <= C_iid` at fixed span.

5. Growing-span degradation.
When the within-window span grows with `n`, the estimated `kappa(Delta_n)` is no longer `O(1)`. That supports a crossover picture: bounded-span windows inherit the root-`n` carrier, while sufficiently fast span growth degrades the effective constant and eventually the observed finite-sample rate.

6. Bounded-support target is cleaner than Gaussian tails.
The Gaussian location model is useful for intuition, but the explicit asymptotic constant is tail-delicate. Numerically, the bounded-support uniform-noise model gives a cleaner fixed-span inheritance picture and is the better immediate theorem target.

## Local theorem target

### Conservative

In the 1-D Gaussian location model with bounded fixed span, prove

`n E W_2^2(hat P_n^{tri}, bar P_n) -> C_tri^2(Delta, sigma)`

for an explicit quantile-process constant `C_tri(Delta, sigma)`.

### Moderate

Show also that

`C_tri(Delta, sigma) <= C_iid(Delta, sigma)`

for the corresponding i.i.d. mixture benchmark, yielding an inheritance theorem with no loss in exponent and no worsening in constant.

## Main open proof ingredient

The remaining theorem-level gap is a Bahadur / empirical-quantile representation for the non-identically distributed triangular array with uniform control strong enough to pass from pointwise quantile fluctuations to the integrated `W_2^2` statement.

For the next proof attempt, bounded support and fixed span should be treated as the primary target assumptions.
At the package level, this note feeds the minimum carrier instantiation; the main theorem object remains the abstract `(a,H)` law.
