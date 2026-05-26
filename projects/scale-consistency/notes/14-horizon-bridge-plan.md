# 14. Horizon Bridge Plan
Status: active
Category: task
Prev: 13. Scale-Consistency V1 Plan

## Scientific object

Infer drift regularity from lag geometry and map it into the temporal-validity horizon.

- observability side: `D_j = zeta j^H`
- information side: `I_{n,L}(H) = n \sum_{j \le L} j^{2H}`
- horizon side: `n_* = (a C_K / (H C_S zeta))^{1 / (a + H)}`

The bridge succeeds when the lag-geometry estimate is accurate enough that the plug-in horizon is operationally meaningful.

## Immediate empirical contract

1. show exact or near-exact recovery under the correctly specified lag law
2. quantify degradation under misspecification
3. only after that, compare plug-in horizons with empirical optima on real streams

## Experiment order

1. E1 synthetic bridge recovery
2. E2 misspecification stress test
3. E3 real-stream lag geometry
4. E5 matched drift pairs
5. E4 horizon versus burden

## E1 outputs

For each `(n, H, zeta, sigma0)` grid point report:

1. bias and RMSE of `\hat H`
2. bias and RMSE of `\hat zeta`
3. bias and RMSE of `\hat n_*`
4. residual slope diagnostic in log-lag space

The main expected signature is monotone error decay as `I_{n,L}(H)` grows.

## E2 outputs

For each misspecification family and amplitude report:

1. bias and RMSE of `\hat H`
2. bias and RMSE of `\hat n_*`
3. residual slope diagnostic

The main expected signature is that horizon bias grows with misspecification amplitude, with oscillatory and piecewise perturbations producing stronger distortion than the exact model.

## Decision rule

Proceed to real streams only if:

1. correct-model bridge recovery is accurate across a nontrivial grid
2. misspecification failure is detectable rather than silent
3. plug-in horizon error remains smaller than the empirical spread of downstream burden quantities on the tested synthetic grid
