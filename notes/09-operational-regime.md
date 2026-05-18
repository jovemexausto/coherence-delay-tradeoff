# 09. Operational Sinkhorn Extension
Status: active
Category: conjecture
Prev: 08. Extended Regime
Next: 10. Paper Next Steps

This note records the operational fixed-`epsilon` Sinkhorn extension in its
current honest form.

## Closed structural ingredients

- iid fixed-`epsilon` benchmark results exist in the literature;
- support-complexity inheritance is exact on the embedded fixed-span model;
- the dual smoothness threshold is `alpha > k/2`.

These ingredients are theorem-level within the model used by the paper.

## Conjecture

The full missing step is triangular-array horizon inheritance:

- `E S_epsilon(\hat P_{t,tri}^{(n)}, \bar P_t^{(n)}) <= C_epsilon n^{-a_epsilon}`

with the same exponent as the iid benchmark on the relevant regularization band.

If that conjecture holds, it feeds the general horizon law exactly as any other
finite-sample exponent does.

## Current evidence

On the tested embedded fixed-span grids:

- triangular and iid slope estimates remain close;
- changing `epsilon` mainly changes constants;
- no qualitative exponent collapse appears on the reported bands.

## Paper status

The paper should present:

- the closed structural ingredients as theorem-level statements;
- the full inheritance claim as a conjecture;
- the calibration grid as empirical evidence only.
