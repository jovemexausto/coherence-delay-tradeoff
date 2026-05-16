# Practical Layer Measurement

This note states Proposition 5 as a measurement-layer theorem target with current empirical support.
Its role is to make the practical layer precise without overstating what is already proved.

## Goal

Obtain a measurement layer whose carrier behavior is stable enough in practice to feed the abstract carrier-roughness useful-memory horizon law.

The current target is fixed-`epsilon` debiased Sinkhorn on low-intrinsic-dimensional support.

## Proposition 5 target

Let `S_{\epsilon}` denote a fixed-`epsilon` debiased Sinkhorn divergence.

The theorem target is:

- under bounded support and fixed span;
- in a regime where the support has low intrinsic dimension;
- the triangular-window carrier under `S_{\epsilon}` has the same effective exponent as the i.i.d. mixture benchmark up to a constant-level gap.

In symbols, the intended form is

`E S_{\epsilon}(\hat P_{t,tri}^{(n)}, \bar P_t^{(n)}) <= C_{tri,\epsilon} n^{-a_\epsilon}`

with the benchmark satisfying

`E S_{\epsilon}(\hat P_{t,iid}^{(n)}, \bar P_t^{(n)}) <= C_{iid,\epsilon} n^{-a_\epsilon}`

for the same effective exponent `a_\epsilon` in the regime under study.

## Why this layer matters

The practical layer is where the abstract law becomes usable beyond the raw `W_2` regimes where dimensional barriers are severe.

The paper does not need this layer to be fully closed as a theorem in order to have a complete central contribution.
But it does need a clear statement of what the practical target is and what the current evidence actually says.

## Current empirical support

The current signal comes from the low-intrinsic-dimensional experiments summarized in `notes/carrier_roughness_research.md`.

On embedded `k = 1` support inside `ambient_dim = 8`, the measured triangular and i.i.d. slopes remain close across `epsilon` values.

Representative signal:

- `epsilon = 0.50`: triangular and i.i.d. slopes are close;
- `epsilon = 0.20`: same qualitative behavior;
- `epsilon = 0.10`: same qualitative behavior;
- `epsilon = 0.05`: same qualitative behavior.

Across those sweeps, the main signature is:

- triangular and i.i.d. exponents stay close;
- changing `epsilon` primarily moves constants;
- there is no clear qualitative phase change across the tested `epsilon` values.

That is the right practical-layer sign, even though it is not yet a proof.

## What should be claimed now

The current paper can honestly claim:

- fixed-`epsilon` Sinkhorn is a plausible practical measurement layer for the abstract law;
- the current sweeps show carrier stability signatures rather than a collapse of exponent;
- the practical layer is empirically consistent with the useful-memory horizon framework.

## What should not be claimed now

The current paper should not claim:

- that fixed-`epsilon` Sinkhorn is already a closed carrier theorem;
- that it universally restores the statistical carrier in arbitrary dimension;
- that `epsilon` has been optimized or characterized sharply.

Those are later measurement-layer results, not current theorems.

## Source map

- `notes/carrier_roughness_research.md`: practical-layer experimental evidence.
- `code/useful_memory_horizon/carrier_roughness_research.py`: experiment harness for the broader carrier sweeps.
