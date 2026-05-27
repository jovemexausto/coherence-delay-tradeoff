# 17. Horizon Duality
Status: active
Category: note
Prev: 16. Validity Partition Ratio
Next: -

## Result

- The validity-partition ratio `R_t(n) = (C_S \zeta_t n^H)/(C_K n^{-a})` is exact at the optimizer: `R_t(n^*) = a/H`.
- `R_t(n)` is strictly increasing in `n`, so `sign(R_t(n_\pi)-a/H)` matches `sign(n_\pi-n_t^*)` and the sign of the excess loss.
- A model-free sign detector using only the observed U-curve can locate the side of `n^*` without estimating `(a,H,\zeta)`, although slope-based recovery of `a/H` can be unstable on sparse grids.

## Interpretation

- Forward law: parameters `\to` horizon.
- Dual law: observed error partition `\to` direction to horizon.
- This closes the constitutive picture with a coordinate on the validity field, not only an optimizer.
