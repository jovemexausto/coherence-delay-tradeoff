# 07. Witness Shape Extremality
Status: active
Category: refinement
Prev: 06. Exact Gaussian Witness Frontier
Next: 08. Extended Regime

The structural question beyond the ramp witness is:

> among endpoint-saturating `H`-Hölder witness shapes, which one minimizes the
> testing energy and therefore gives the strongest Gaussian two-point lower bound?

The answer is clean.

## Setup

Fix `H in (0,1]` and an integer width `h >= 1`. Consider profiles

`g = (g_0, ..., g_h)`

such that:

- `g_0 = 0`;
- `g_h = h^H`;
- `|g_r - g_s| <= |r-s|^H` for all `0 <= s <= r <= h`.

These are exactly the discrete endpoint-saturating `H`-Hölder shapes with unit
roughness budget.

For the Gaussian two-point witness, the endpoint signal is fixed by `g_h`, while
the testing difficulty depends on the energy

`E(g) = sum_{r=1}^h g_r^2`.

So the strongest witness in this class is the one with the smallest feasible
energy.

## Extremal lower-envelope profile

For every feasible profile `g` and every `r`, the endpoint Hölder constraint gives

`g_r >= h^H - (h-r)^H`.

Therefore the pointwise smallest feasible profile is

`g_r^{min} = h^H - (h-r)^H`.

This profile is itself feasible because for `r >= s`,

`g_r^{min} - g_s^{min} = (h-s)^H - (h-r)^H <= (r-s)^H`

by subadditivity of `x -> x^H` on `(0, infinity)` for `H <= 1`.

Consequently:

- `g^{min}` is feasible;
- every other feasible profile dominates it pointwise;
- hence `g^{min}` minimizes `E(g)` over the whole endpoint-saturating class.

So the ramp `g_r = r^H` is not extremal for `H < 1`.

At `H = 1`, the two coincide:

`h - (h-r) = r`.

That is why the Lipschitz endpoint is special.

## Energy constants

For the ramp witness,

`E_ramp(h) = sum_{r=1}^h r^{2H} ~ h^{2H+1} / (2H+1)`.

For the endpoint-minimal witness,

`E_min(h) = sum_{r=1}^h (h^H - (h-r)^H)^2`.

After rescaling by `r = hu`,

`E_min(h) ~ I_H h^{2H+1}`

with

`I_H = int_0^1 (1 - (1-u)^H)^2 du = 2H^2 / ((H+1)(2H+1))`.

Since

`2H^2 / ((H+1)(2H+1)) < 1 / (2H+1)` for `H < 1`,

the endpoint-minimal witness has strictly smaller asymptotic energy than the ramp.

## Consequence for the exact Gaussian frontier

For any profile family with endpoint `h^H` and energy constant `I`, the exact
Gaussian asymptotic two-point reduction gives

`C_H(I) = I^{-H/(2H+1)} x_H^{2H/(2H+1)} Phi(-x_H)`

where `x_H` solves

`p_H Phi(-x_H) = x_H phi(x_H)`, with `p_H = 2H / (2H+1)`.

Applying this to the endpoint-minimal profile gives

`C_H^{min} = I_H^{-H/(2H+1)} x_H^{2H/(2H+1)} Phi(-x_H)`

with

`I_H = 2H^2 / ((H+1)(2H+1))`.

This strictly improves the ramp constant for `H < 1`, with improvement factor

`( (H+1) / (2H^2) )^{H/(2H+1)}`.

## Numerical status

The module `code/useful_memory_horizon/gaussian_witness_frontier.py` and its test
suite verify four things.

1. The exact ramp frontier matches its asymptotic constant.
2. The endpoint-minimal profile has the predicted energy constant.
3. The endpoint-minimal profile strictly beats the ramp for `H < 1`.
4. The discrete optimum for the endpoint-minimal profile matches its predicted
   asymptotic constant and horizon shape.

Consequences for the lower theory:

- the lower-bound exponent remains the same;
- the ramp witness remains a valid structural witness;
- but it is not the strongest endpoint-saturating Hölder witness except at
  `H = 1`.
