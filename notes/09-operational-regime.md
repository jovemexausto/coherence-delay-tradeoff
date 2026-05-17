# 09. Operational Regime
Status: active
Category: regime
Prev: 08. Extended Regime
Next: 10. Paper Next Steps

Operational-regime carrier theorem in its intended form, with the current
empirical support made explicit.

## Goal

Obtain a measurement geometry whose carrier behavior is stable enough to feed
the general carrier-roughness horizon law in regimes where raw `W_2` is too
fragile or too dimension-sensitive.

The current target is fixed-`epsilon` debiased Sinkhorn on low-intrinsic-dimensional support.

## Operational-regime theorem form

Let `S_{\epsilon}` denote a fixed-`epsilon` debiased Sinkhorn divergence.

The intended theorem form is:

- under bounded support and fixed span;
- in a regime where the support has low intrinsic dimension;
- the triangular-window carrier under `S_{\epsilon}` has the same effective exponent as the i.i.d. mixture benchmark up to a constant-level gap.

In symbols, the intended form is

`E S_{\epsilon}(\hat P_{t,tri}^{(n)}, \bar P_t^{(n)}) <= C_{tri,\epsilon} n^{-a_\epsilon}`

with the benchmark satisfying

`E S_{\epsilon}(\hat P_{t,iid}^{(n)}, \bar P_t^{(n)}) <= C_{iid,\epsilon} n^{-a_\epsilon}`

for the same effective exponent `a_\epsilon` in the regime under study.

## Why this regime belongs to the paper

The operational regime is where the general law becomes usable beyond raw `W_2`
regimes where dimensional barriers are severe.

The paper does not need this regime to be fully closed in order to have a
complete central contribution. But it does need a clear theorem form and a clear
statement of what the present evidence actually supports.

## Current empirical support

On embedded `k = 1` support inside `ambient_dim = 8`, the measured triangular and i.i.d. slopes remain close across `epsilon` values.

Representative signal:

- `epsilon = 0.50`: triangular and i.i.d. slopes are close;
- `epsilon = 0.20`: same qualitative behavior;
- `epsilon = 0.10`: same qualitative behavior;
- `epsilon = 0.05`: same qualitative behavior.

In a larger sweep on `ambient_dim = 8`, `intrinsic_dim = 1`, the same pattern
appears numerically as:

- `epsilon = 0.50` -> triangular `a \approx 0.460`, i.i.d. `a \approx 0.527`;
- `epsilon = 0.20` -> triangular `a \approx 0.459`, i.i.d. `a \approx 0.531`;
- `epsilon = 0.10` -> triangular `a \approx 0.451`, i.i.d. `a \approx 0.534`;
- `epsilon = 0.05` -> triangular `a \approx 0.449`, i.i.d. `a \approx 0.534`.

Across those sweeps, the main signature is:

- triangular and i.i.d. exponents stay close;
- changing `epsilon` primarily moves constants;
- there is no clear qualitative phase change across the tested `epsilon` values.

That is the right operational-regime sign, even though it is not yet a proof.

## Current status

This is the intended full theorem form for the operational regime.
At present it remains a theorem-shaped regime statement supported by evidence,
not a closed theorem.

## What should be claimed now

The current paper can honestly claim:

- fixed-`epsilon` Sinkhorn is a plausible operational geometry for the general law;
- the current sweeps show carrier stability signatures rather than a collapse of exponent;
- the operational regime is empirically consistent with the useful-memory horizon framework.

## What should not be claimed now

The current paper should not claim:

- that fixed-`epsilon` Sinkhorn is already a closed carrier theorem;
- that it universally restores the statistical carrier in arbitrary dimension;
- that `epsilon` has been optimized or characterized sharply.

Those are later operational-regime results, not current theorems.

## Source map

- `code/useful_memory_horizon/carrier_roughness_research.py`: experiment harness for the broader carrier sweeps.
