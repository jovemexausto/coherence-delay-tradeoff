# 12. Theorem Ladder
Status: active
Category: roadmap
Prev: 11. Future Work and Open Problems
Next: -

Routing map for the next theorem moves. Start from the now-closed paper core and
order the remaining work by scientific leverage.

## Core

Already in hand:

- abstract upper law;
- optimized horizon law;
- exact finite-`n` uniform-window staleness constant;
- minimum kernel;
- structural lower bound on the canonical `a = 1/2` regime.

These are not the current bottleneck.

## First move: promote the lower-regime refinements

The next theorem-ready target is no longer a new exponent. It is a sharper and
cleaner asymptotic proposition on the canonical lower regime.

### Target 1A: Exact Gaussian ramp frontier

Write the exact Gaussian two-point reduction in asymptotic proposition form:

- derive the scalar root equation `p_H Phi(-x_H) = x_H phi(x_H)`;
- derive `C_H^{ramp}` and `A_H^{ramp}`;
- justify the discrete-to-asymptotic passage carefully.

Reference note: `notes/06-exact-gaussian-witness-frontier.md`.

### Target 1B: Witness-shape extremality

Write the endpoint-minimal witness result in proposition form:

- characterize the feasible endpoint-saturating Hölder class;
- prove the pointwise lower envelope `g_r^{min} = h^H - (h-r)^H`;
- compute the energy constant `I_H = 2H^2 / ((H+1)(2H+1))`;
- derive the stronger constant `C_H^{min}`.

Reference note: `notes/07-witness-shape-extremality.md`.

This is the highest-leverage next theorem move because it strengthens the canonical
lower regime without changing the underlying theory.

## Second move: extended-regime carrier theorem

Upgrade beyond the proof kernel while preserving the carrier-roughness law.

Target:

- low-dimensional / low-intrinsic-dimensional fixed-span carrier inheritance.

Reference note: `notes/08-extended-regime.md`.

This is the cleanest broader carrier theorem beyond the minimum kernel.

## Third move: operational-regime carrier theorem

Move to a measurement geometry that is robust enough for practice.

Target:

- fixed-`epsilon` Sinkhorn or another dimension-robust measurement geometry with a
  theorem-level carrier bound feeding the same upper law.

Reference note: `notes/09-operational-regime.md`.

## Fourth move: beyond-canonical-regime lower theory

The lower frontier beyond `a = 1/2` now has a clearer formulation than before.

Targets:

- build carrier-matched witnesses for `a != 1/2`;
- determine whether the endpoint-minimal witness is optimal inside a larger class;
- move from subclass lower bounds toward class-tight lower bounds.

This is important, but it is no longer the first thing to do next.

## What should stop consuming time

- searching for a different exponent on the canonical `a = 1/2` regime;
- treating the ramp witness as if it were automatically the best shape for `H < 1`;
- trying to close the entire `(a,H)` family before the refined lower regime is
  written cleanly;
- pretending the operational regime is theorem-level before it is.

## Current priority order

1. Write the exact Gaussian ramp frontier cleanly.
2. Write the witness-shape extremality refinement cleanly.
3. Decide which part enters the paper body and which part stays in appendix/notes.
4. Return to the extended-regime theorem.
5. Then return to the operational-regime theorem.
