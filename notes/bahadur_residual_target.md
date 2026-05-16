# Bahadur Residual Target for P2

This note tracks the remainder term that closes the minimum kernel.
Its scope is deliberately local to Proposition 3 in `notes/main_theorem_package.md`.

## What to check next

The remaining theorem ingredient should be a uniform control of the remainder

`R_n(u) = \hat q_n(u) - q_n(u) - (u - \hat F_n(q_n(u))) / \bar f_n(q_n(u))`

such that, at least on an interior quantile band `u \in [\varepsilon, 1-\varepsilon]`,

`\int_{\varepsilon}^{1-\varepsilon} E[R_n(u)^2] du = o(n^{-1})`.

## Exploratory diagnostics

For bounded-support fixed-span windows, measure both the full grid and an interior band:

- `integral residual`
- `integral residual / integral mse`
- residual rate in `n`

The interior-band version is the more relevant proxy for the proof, because the theorem assumptions only need density control away from the boundaries.

The target behavior is:

- residual fraction small for moderate `n`
- residual rate larger than `1/2`
- same qualitative behavior for triangular and i.i.d. mixture designs

## Current numerical status

On the fixed-span bounded-support sweep, the integrated residual fraction drops with `n`, and the sup-remainder rate is comfortably above `1/2` in both designs. The interior-band version is the cleaner target and should be the one to cite in the proof sketch.

The empirical increment term is the rate bottleneck; the Taylor term can be smoother, but the full remainder should be stated at the classical `o(n^{-1/2})` level rather than as a pure Taylor rate.

## Why this is the right next step

If the residual is controlled, the minimum theorem follows from the leading quantile linearization plus the variance calculation already observed in the Bahadur diagnostics.
