# Bahadur Frontier Map

## Safe zone

- 1-D bounded support
- fixed span
- interior quantile band `u in [epsilon, 1-epsilon]`
- density bounded below and uniformly Hölder on the interior

In this regime, the lab supports:

- `sup_u |R_n(u)| = o_p(n^{-1/2})`
- `int E[R_n(u)^2] du = o(n^{-1})`
- root-`n` carrier inheritance for `W_2`

## Boundary 1: span growth

When the span grows like `base_span * n^beta`, the effective exponent degrades.

Empirically:

- small `beta` is still safe;
- by `beta ~ 0.75` the remainder weakens;
- by `beta = 1.0` the root-`n` story breaks clearly.

## Boundary 2: roughness alone

Replacing the kernel with a cusp density changes the Taylor term, but by itself it does not destroy the root-`n` remainder in the fixed-span interior-band lab.

## Real failure mode

The first genuine failure is the combined roughness + span-growth regime.

That is the boundary where the empirical increment and the geometry of the moving window stop supporting the fixed-span proof.

## Proof takeaway

The minimum theorem is closed in the safe zone.
The theorem should not be stated beyond fixed span unless the span-growth degradation is built into the conclusion.
