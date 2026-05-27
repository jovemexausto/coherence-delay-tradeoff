# 15. Signed Residual Baseline
Status: active
Category: note
Prev: 14. Checkpoint
Next: -

## Finding

- In a reduced calibrated frontier over `observation`, `signed_residual`, and `absolute_residual`, the signed transform did not improve the validity-detection gap.
- `signed_residual` lowered the mean gap aggregate relative to `observation` (`~256` vs `~291`) and kept the same weak positive-gap rate (`~0.375`).
- `absolute_residual` remained more conservative (`~335` mean gap) but did not improve positivity over `observation`.

## Implication

- Residual transformations do not remove the validity-detection lag on the current benchmark.
- `observation` remains the best baseline input for the calibrated frontier.
- The signed transform is a robustness check, not a cure.
