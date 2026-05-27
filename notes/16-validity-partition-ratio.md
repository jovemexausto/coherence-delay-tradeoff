# 16. Validity Partition Ratio
Status: active
Category: note
Prev: 15. Signed Residual Baseline
Next: -

## Claim

- The horizon law admits a conservation-style reading at the optimizer: the finite-sample and staleness terms are balanced by the first-order condition, and the ratio of the two terms equals `a/H` at `n*`.
- This suggests a dimensionless validity-partition ratio `R_t(n) = (C_S \zeta n^H)/(C_K n^{-a})` as a dual coordinate for locating the operating point relative to the horizon.

## Caveat

- The ratio is theorem-clean in the benchmark model where the two components are separately identifiable.
- Outside that model, the decomposition is only as good as the error partition being estimated, so the object is a candidate self-calibrating coordinate, not yet a fully model-free observable.

## Use

- Candidate bridge from horizon geometry to online control.
- Candidate alternative to lag-geometry estimation when the component decomposition is available directly.
