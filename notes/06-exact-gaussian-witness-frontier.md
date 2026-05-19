# 06. Exact Gaussian Ramp Frontier
Status: closed
Category: refinement
Prev: 05. Structural Lower Theory
Next: 07. Endpoint-Minimal Profile

This note records the exact Gaussian two-point calculation for the ramp profile.

## Setup

Use the Gaussian path pair

- `mu_{t-j}^{\pm,h} = \pm beta (h-j)_+^H`.

For fixed `h`, the exact two-point lower payoff is

- `L_h(beta) = beta h^H Phi(- beta sqrt(S_h) / sigma)`

with `S_h = sum_{r=1}^h r^{2H}`.

## Asymptotic frontier

Let

- `p_H = 2H / (2H+1)`.

After the large-ratio scaling `sigma / zeta -> infinity`, the normalized ramp
problem reduces to maximizing

- `x^{p_H} Phi(-x)`.

The maximizer `x_H` solves

- `p_H Phi(-x_H) = x_H phi(x_H)`.

This yields the asymptotic constant

- `C_H^{\mathrm{ramp}} = (2H+1)^{H/(2H+1)} x_H^{2H/(2H+1)} Phi(-x_H)`.

## Status

This refinement is now incorporated into the paper through the compact refined
Gaussian constants proposition and the appendix proof details.
