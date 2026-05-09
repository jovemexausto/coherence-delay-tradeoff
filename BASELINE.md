# Baseline

## Scientific Core
- The paper's governing object is the coherence-delay trade-off.
- Finite-memory tracking under drift has a cube-root horizon law.
- Temporal coherence means operational alignment with the current distribution.
- Freshness is only a rate-free proxy for memory age, not the main object.
- The relevant objects are horizon geometry, horizon tracking, and predictive payoff.
- Detector timing and memory-validity timing are different objects.
- Horizon misalignment induces a nonlinear operational cost.
- That cost is method-dependent and regime-dependent.

## What Is Supported
- `U`-curve under drift.
- Cube-root law for useful memory.
- Oracle horizon recovery by phase.
- `cap-only` regime.
- Lag-variance frontier.
- Temporal-validity gap.
- Horizon cost curve.
- Alternating-timescales instability.

## What Is Not Supported
- Universal MAE dominance of the cube-root regulator.
- Claims that following the oracle horizon always improves prediction.
- Treating a representative trace as aggregate evidence.

## Current Interpretation
- The paper characterizes the temporal geometry of useful memory under drift.
- The cube-root law is the operational scale induced by the coherence-delay trade-off.
- The main result is a separation between statistical detectability and temporal validity.
- Operational hysteresis appears as asymmetric contraction/re-expansion of the useful horizon.

## Rules For Next Work
- Do not return to a leaderboard framing.
- Do not collapse horizon tracking into predictor optimality.
- Do not let `freshness`, `validity`, and `coherence` become interchangeable.
- Treat new experiments as tests of a transfer function: gap to cost.
- Prefer diagnostics that separate regime, backend, and temporal scale.
