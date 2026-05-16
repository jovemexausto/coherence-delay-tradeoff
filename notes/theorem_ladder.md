# P2 Theorem Ladder

This note is a routing map for the theorem package in `notes/main_theorem_package.md`.
The ladder exists to supply carrier instantiations for the temporal-validity law, not to define a detector rule or a regret-tuning scheme.

## Minimum

Prove the fixed-span triangular-array carrier under bounded support and uniform density control.

Target:

`E W_2(\hat P_n^{tri}, \bar P_n) = O(n^{-1/2})`

for independent, non-identical samples in a window, with the population target `\bar P_n` given by the mixture of the windowed laws.

The theorem-ready statement now lives in `notes/minimum_kernel_proposition.md`.
The proof narrative now lives in `notes/minimum_kernel_proof.md`.

## Useful

Upgrade from the minimum kernel to low-dimensional / low-intrinsic-dimensional settings and show that the triangular array inherits the i.i.d. benchmark carrier exponent `a` whenever that benchmark is known.

The focused theorem target is now in `notes/useful_layer_bridge.md`.

## Practically relevant

Move to a measurement layer that is dimension-robust in practice, e.g. fixed-`\epsilon` Sinkhorn, or to raw `W_2` under explicit intrinsic-dimension assumptions.

The focused measurement-layer target is now in `notes/practical_layer_measurement.md`.

The current useful-layer bridge is the fixed-span low-intrinsic carrier inheritance theorem, which is the moderate theorem targeted by the current experiments.

## Current research signal

- Fixed-span inheritance is numerically stable.
- Growing span degrades the effective constant and the observed rate.
- The cleanest theorem target is bounded support first, Gaussian tails second.
- The main paper contribution should be organized through the abstract `(a,H)` law, with the ladder supplying carrier instantiations.
- The abstract paper-level theorem is now separated cleanly in `notes/abstract_upper_law.md`.
- The package boundary note is `notes/package_boundaries.md`.

## Proof skeleton for the minimum theorem

1. Represent `W_2` through quantiles in the minimum kernel setting.
2. Use a Bahadur-type expansion for the triangular-array empirical quantile process.
3. Show the leading term has variance `O(1/n)` uniformly in `u` under bounded support and fixed span.
4. Control the remainder uniformly so that it is `o(n^{-1/2})` after integration.
5. Integrate over `u` to obtain `E W_2^2 = O(1/n)` and therefore `E W_2 = O(n^{-1/2})`.
6. Compare the triangular-array variance constant with the i.i.d. mixture constant to identify the design effect.

At this point the main remaining task is consolidation into one proof narrative, not searching for a different exponent.

## What could break the proof

- Vanishing density near the support boundary.
- Lack of uniform regularity in the component laws.
- Span growing with `n`.
- Remainder terms in the Bahadur representation not uniform in `u`.
- Roughness plus span growth together can push the remainder below the root-`n` barrier.
