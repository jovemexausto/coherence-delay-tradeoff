# Working Plan

## Current object

Define on the same filtered stream:

1. `tau_valid(n)`: the first time a fixed operating window `n` exits the temporal-validity region.
2. `tau_detect`: the first alarm time of a sequential detector run under a false-alarm constraint.
3. `Delta_val-det = tau_detect - tau_valid(n)`.

The main empirical question is whether `Delta_val-det` remains positive after calibrating detector aggressiveness on the same null false-alarm budget.

The theorem-level false-alarm notion is the finite-horizon budget `P_0(tau <= T) <= alpha` under the stationary Gaussian null.

## Compact experimental contract

1. Use false-alarm-calibrated detector frontiers rather than raw parameter sweeps whenever detectors are compared.
2. Keep observation-input baselines front-stage: `ADWIN`, `PageHinkley`, `CUSUM`, and `KSWIN`.
3. Keep residual-input variants as scope diagnostics, since they often trade shorter nominal reaction rules for missed alarms or much larger delays.
4. Run the frontier for `H in {0.5, 0.75, 1.0}` so the phenomenon is not tied to the Lipschitz case.

## General theorem target

The clean target is a detector-class sign theorem, not a detector-specific tuning statement.

### Candidate theorem

For a class of adapted stopping times satisfying a null false-alarm constraint and a finite-information accumulation bound, there exists a Hölder drift family with exponent `H in (0, 1]` such that

`P(tau_detect > tau_valid(n)) >= c`

for some `c > 0`, uniformly over a range of operating windows `n` whose horizon has already collapsed.

### Current benchmark theorem

For the one-dimensional Gaussian location model with a null prefix through the validity-expiry time, every `tau in D_{alpha,T}` satisfies

`P(tau_detect > tau_valid(n)) >= 1 - alpha`.

This is the first sign result in the line.

### Stronger target

Under the same class conditions, prove a lower bound of the form

`E[(tau_detect - tau_valid(n))_+] >= c * g(alpha, zeta, H, n)`

where `alpha` is the false-alarm budget and `g` is positive in the pre-detection invalidity regime.

## Proof route

1. Start from a one-dimensional Gaussian location model with piecewise Hölder mean path and fixed observation noise.
2. Define `tau_valid(n)` from the closed horizon law `n^*(a, H) ~ (C_K / zeta)^{1/(a+H)}` with `a = 1/2` in the benchmark model.
3. Show that after the validity horizon is crossed, the post-change information accumulated over a short interval is still below what is needed for reliable alarm under the false-alarm constraint.
4. Convert that information deficit into a lower bound on alarm delay using a sequential testing inequality.
5. Separate theorem-level detector-class assumptions from benchmark detector instantiations.

## Current empirical signal

The compact calibrated frontier already supports the theorem target qualitatively.

1. Observation-based `ADWIN`, `PageHinkley`, and `CUSUM` retain positive mean gaps on the tested `H in {0.5, 0.75, 1.0}` grid.
2. `KSWIN` also shows positive gaps but with materially lower detection rates, so it is better used as a robustness baseline than as the main benchmark family.
3. Residual-input variants often produce much larger delays or missed alarms, which supports a scope statement: input representation can worsen the validity-detection gap instead of curing it.

## Writing rule

The paper should treat sequential validity versus detection as a second statistical clock layered on top of the temporal-validity horizon, not as an appendix-style operational afterthought.
