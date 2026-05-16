# Useful Layer Bridge

This note states Proposition 4 in a focused theorem-target form.
Its role is to bridge the minimum kernel to the useful layer without pretending that the full high-dimensional carrier problem is already solved.

## Goal

Show that, under bounded support, fixed span, and low intrinsic dimension, the triangular-array carrier inherits the same exponent as the i.i.d. mixture benchmark up to a small constant-level gap.

The point is not to prove dimension-free raw `W_2` behavior in full generality.
The point is to identify a scientifically relevant regime where the minimum-kernel carrier survives beyond the one-dimensional proof kernel.

## Setup

At time `t`, let:

- `\bar P_t^{(n)}` be the window mixture target;
- `\hat P_{t,tri}^{(n)}` be the empirical law of the triangular window;
- `\hat P_{t,iid}^{(n)}` be an i.i.d. sample of size `n` from `\bar P_t^{(n)}`.

Assume the support of `\bar P_t^{(n)}` is embedded in ambient dimension `d` but has low intrinsic dimension `k`.

## Proposition 4 target

Under bounded support, fixed span, and low intrinsic dimension, if the i.i.d. mixture benchmark satisfies a carrier law

`E W_2(\hat P_{t,iid}^{(n)}, \bar P_t^{(n)}) <= C_iid n^{-a}`

then the triangular window should satisfy

`E W_2(\hat P_{t,tri}^{(n)}, \bar P_t^{(n)}) <= C_tri n^{-a}`

with the same exponent `a` and only a constant-level design gap.

In the main useful slice, the benchmark exponent should remain near `a = 1/2`.

## Why this is the right theorem target

This is the first layer where the paper becomes scientifically broader than the minimum kernel.

The theorem does not need to solve the full high-dimensional `W_2` problem. It only needs to show that in the regime where the benchmark carrier is still useful, the triangular design does not destroy that exponent.

That is enough to make the abstract `(a,H)` law useful outside the one-dimensional proof kernel.

## Current numerical support

The strongest stable signal is the embedded `k = 1` case under fixed span.

Current practical thresholds supported by the lab and encoded in `tests/test_glue_theorem_useful.py` are:

- `a_tri > 0.40`;
- `a_iid > 0.40`;
- `|a_tri - a_iid| < 0.15`.

The stricter aspirational target used in the research notes is:

- `a_tri > 0.45`;
- `a_iid > 0.45`;
- `|a_tri - a_iid| < 0.08`.

This gap between test threshold and aspirational threshold is fine at this stage. The test is meant to guard the qualitative theorem target, not to lock in a brittle numerical constant.

## Current experiment family

The current useful-layer lab in `code/useful_memory_horizon/glue_theorem_useful.py` tests:

- embedded low-intrinsic supports inside larger ambient spaces;
- fixed-span windows;
- direct slope comparison between triangular and i.i.d. mixture designs.

The core qualitative finding is:

- low intrinsic dimension preserves a carrier near the `a = 1/2` slice;
- the triangular slope tracks the i.i.d. slope closely;
- ambient dimension alone is not the main barrier once intrinsic dimension stays low.

## What remains open

This note does not claim:

- a proof for arbitrary high-dimensional supports;
- a proof for growing-span windows;
- a theorem for raw `W_2` in regimes where the i.i.d. benchmark itself loses the `a = 1/2` carrier.

Those are outside the first useful-layer bridge.

## Source map

- `notes/carrier_roughness_research.md`: broader research context and numerical evidence.
- `code/useful_memory_horizon/glue_theorem_useful.py`: current experiment harness.
- `tests/test_glue_theorem_useful.py`: regression test for the qualitative bridge signal.
