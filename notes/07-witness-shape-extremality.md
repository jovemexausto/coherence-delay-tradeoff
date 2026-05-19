# 07. Endpoint-Minimal Profile
Status: closed
Category: refinement
Prev: 06. Exact Gaussian Ramp Frontier
Next: 08. Extended Regime

Endpoint-minimal profile sharpening the Gaussian lower
constants for `H < 1`.

## Extremal profile

Among endpoint-saturating discrete Holder profiles,

- `g_r^{min} = h^H - (h-r)^H`

is the pointwise minimal feasible profile and therefore minimizes testing energy.

## Energy constant

Its asymptotic energy constant is

- `I_H = 2H^2 / ((H+1)(2H+1))`.

Substituting this into the exact Gaussian two-point reduction yields

- `C_H^{\min} = I_H^{-H/(2H+1)} x_H^{2H/(2H+1)} Phi(-x_H)`.

For `H < 1`, the improvement factor over the ramp constant is

- `((H+1)/(2H^2))^{H/(2H+1)}`.

## Status

This lower-bound geometry is now incorporated into the paper through the compact
Gaussian constants proposition and the appendix proof details.
