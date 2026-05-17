# 08. Extended Regime
Status: active
Category: regime
Prev: 07. Witness Shape Extremality
Next: 09. Operational Regime

Extended-regime carrier theorem in focused form.

This regime captures a broader setting in which the same carrier-roughness law
should persist beyond the minimum-kernel theorem, without pretending that the
full high-dimensional carrier problem is already solved.

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

## Extended-regime theorem

Under bounded support, fixed span, and low intrinsic dimension, if the i.i.d. mixture benchmark satisfies a carrier law

`E W_2(\hat P_{t,iid}^{(n)}, \bar P_t^{(n)}) <= C_iid n^{-a}`

then the triangular window should satisfy

`E W_2(\hat P_{t,tri}^{(n)}, \bar P_t^{(n)}) <= C_tri n^{-a}`

with the same exponent `a` and only a constant-level design gap.

In the canonical low-dimensional regime, the benchmark exponent should remain near `a = 1/2`.

## Why this regime belongs to the paper

This is the first regime where the carrier theory becomes scientifically broader
than the minimum kernel.

The theorem does not need to solve the full high-dimensional `W_2` problem. It
needs to show that in a regime where the i.i.d. benchmark carrier is still
present, the triangular design does not destroy that exponent.

That is enough to make the general law operative outside the minimum-kernel theorem.

## Current numerical support

The strongest stable signal is the embedded `k = 1` case under fixed span.

Representative sweeps support the same picture:

- in `ambient_dim = 8`, intrinsic `k = 1` gives carrier `a \approx 0.47-0.50`;
- in `ambient_dim = 8`, intrinsic `k = 2` gives carrier `a \approx 0.45`;
- a full `d = 8` ambient cube is visibly slower.

A larger sweep with larger `n` makes the same point cleaner:

- `ambient_dim = 4`, `intrinsic_dim = 1` gives triangular raw-`W_2` carrier `a \approx 0.46-0.48`;
- `ambient_dim = 8`, `intrinsic_dim = 1` gives `a \approx 0.47-0.48`;
- `ambient_dim = 8`, `intrinsic_dim = 2` gives `a \approx 0.42-0.43`.

The key empirical signature is that the `k=1` embedded support behaves similarly
across ambient dimensions, while increasing intrinsic dimension slows the
carrier.

Current practical thresholds supported by the lab and encoded in `tests/test_glue_theorem_useful.py` are:

- `a_tri > 0.40`;
- `a_iid > 0.40`;
- `|a_tri - a_iid| < 0.15`.

The stricter aspirational target used in the research notes is:

- `a_tri > 0.45`;
- `a_iid > 0.45`;
- `|a_tri - a_iid| < 0.08`.

This gap between test threshold and aspirational threshold is fine at this stage. The test is meant to guard the qualitative theorem form, not to lock in a brittle numerical constant.

## Current experiment family

The current extended-regime lab in `code/useful_memory_horizon/glue_theorem_useful.py` tests:

- embedded low-intrinsic supports inside larger ambient spaces;
- fixed-span windows;
- direct slope comparison between triangular and i.i.d. mixture designs.

The core qualitative finding is:

- low intrinsic dimension preserves a carrier near the canonical `a = 1/2` regime;
- the triangular slope tracks the i.i.d. slope closely;
- ambient dimension alone is not the main barrier once intrinsic dimension stays low.

## Current status

This is the intended full theorem form for the extended regime.
It is not yet closed at the same level as the minimum-kernel result.

## What remains open

This note does not claim:

- a proof for arbitrary high-dimensional supports;
- a proof for growing-span windows;
- a theorem for raw `W_2` in regimes where the i.i.d. benchmark itself loses the `a = 1/2` carrier.

Those are outside the present extended-regime theorem.

## Source map

- `code/useful_memory_horizon/glue_theorem_useful.py`: current experiment harness.
- `tests/test_glue_theorem_useful.py`: regression test for the qualitative extended-regime signal.
