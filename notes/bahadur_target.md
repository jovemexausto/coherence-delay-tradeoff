# Bahadur Target for P2

## Statement to test

For bounded support and fixed span in 1-D, the triangular-array empirical quantile process should admit a Bahadur-type linearization with `o(n^{-1/2})` integrated remainder.

## Diagnostic quantities

For each `u` on a quantile grid:

`hat q(u) - q(u) = (u - hat F(q(u))) / f(q(u)) + R_n(u)`

The exploratory targets are:

- `integral bias^2`
- `integral variance`
- `integral MSE`
- `integral residual = MSE - bias^2 - variance`

## What should happen if the proof path is right

- `integral MSE` scales like `n^{-1}`
- `integral bias^2` is lower order or at least not larger than `integral variance`
- the residual stays small on the discretized grid
- triangular and i.i.d. curves have the same exponent

## Why this matters

If this diagnostic is stable, the next theorem attempt can focus on a uniform Bahadur remainder plus a quantile-process CLT rather than a more abstract transport argument.

## Formal proposition target

Let `X_1, ..., X_n` be independent, non-identically distributed random variables in one dimension with common bounded support and with component densities uniformly bounded above and below on the interior quantile band.

Let `\hat q_n(u)` be the empirical quantile function of the triangular array and `q_n(u)` the quantile function of the window mixture `\bar P_n`.

The theorem target is a uniform Bahadur linearization of the form

`\hat q_n(u) - q_n(u) = (u - \hat F_n(q_n(u))) / \bar f_n(q_n(u)) + R_n(u)`

with

`\int_0^1 E[R_n(u)^2] du = o(n^{-1})`.

If this holds, then integrating the leading term gives

`E W_2^2(\hat P_n^{tri}, \bar P_n) = C^2 / n + o(n^{-1})`

for an explicit design-dependent constant `C^2`.

This is the cleanest route to the minimum theorem.
