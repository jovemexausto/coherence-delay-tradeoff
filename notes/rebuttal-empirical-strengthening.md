# Empirical Strengthening Note

## Reviewer concern

> Empirical evaluation is primarily qualitative (shapes and scaling) with limited datasets and baselines; stronger comparisons to common adaptive-memory strategies (e.g., EWMA/forgetting factors, adaptive Kalman, time-varying bandwidth selection) would strengthen claims.

## Response core

The empirical section now contains two concrete strengthenings.

1. The online-adaptation sweep now compares the structural controller not only to the raw plug-in rule and the activity baseline, but also to adaptive EWMA and an adaptive scalar Kalman filter on the same five profile families (`default`, `smooth`, `rough`, `alternating`, `ramp_up`) and the same seed grid.
2. The validity-versus-detection experiment now includes a false-alarm-calibrated compact frontier over `H in {0.5, 0.75, 1.0}` and detector baselines `ADWIN`, `Page--Hinkley`, `KSWIN`, and `CUSUM`, instead of relying only on the earlier `H=1` two-detector sweep.

## Quantitative online-baseline summary

Across the compact online sweep, mean error relative to oracle is:

- structural: `1.18`--`1.36`
- adaptive EWMA: `1.28`--`1.64`
- adaptive scalar Kalman: `1.21`--`1.58`
- activity baseline: `1.34`--`1.65`
- raw plug-in rule: `2.30`--`4.41`

The structural controller beats EWMA on `13/15` runs, Kalman on `11/15`, activity on `13/15`, and the raw plug-in rule on `15/15`. The clearest gains occur on the rough and alternating profiles, while the smooth default profile remains the most competitive regime for the classical forgetting-factor baselines.

## Quantitative sequential-delay summary

The false-alarm-calibrated compact frontier shows that the validity-detection gap is not confined to the Lipschitz case.

- Observation-based `ADWIN`, `Page--Hinkley`, and `CUSUM` retain positive mean gaps on the tested `H in {0.5, 0.75, 1.0}` grid.
- On successful validity-expiry runs, mean-gap ranges are approximately:
  - `ADWIN`: `204`--`660`
  - `Page--Hinkley`: `204`--`615`
  - `CUSUM`: `78`--`232`
- Observation-based `KSWIN` also preserves positive gaps when it detects, but with materially lower detection rates and frequent missed alarms.
- Residual-input variants often increase delay and can miss alarms entirely, so they do not remove the validity-detection separation.

## Suggested rebuttal paragraph

We strengthened the empirical section in two directions. First, the online memory-selection sweep now compares the structural controller against adaptive EWMA and an adaptive scalar Kalman filter, in addition to the raw plug-in and activity baselines. Across the tested `default/smooth/rough/alternating/ramp_up` profiles, the structural controller remains best overall, with mean error `1.18`--`1.36` times oracle risk versus `1.28`--`1.64` for EWMA, `1.21`--`1.58` for adaptive Kalman, `1.34`--`1.65` for the activity baseline, and `2.30`--`4.41` for the raw plug-in rule; it beats EWMA on `13/15` runs and Kalman on `11/15`, with the clearest gains in rough and alternating regimes. Second, the validity-versus-detection experiment now uses a false-alarm-calibrated compact frontier over `H in {0.5,0.75,1.0}` and detectors `ADWIN`, `Page--Hinkley`, `KSWIN`, and `CUSUM`. On the observation input, `ADWIN`, `Page--Hinkley`, and `CUSUM` all retain strictly positive mean validity-detection gaps across the tested compact grid, while `KSWIN` and residual-input variants often trade larger delays for missed alarms. These additions make the empirical claims less qualitative and broaden the baseline coverage substantially.
