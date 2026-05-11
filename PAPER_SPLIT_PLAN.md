# Paper Split Plan

## Paper 1: Conservative, Publishable Result

### Core claim
- Finite-memory tracking under $W_2$-Lipschitz drift has a structural worst-case floor.
- The cube-root law is the correct minimax story for the ramp-like / adversarial regime.

### Keep
- Variance + staleness decomposition.
- Upper bound for the finite-memory averaging class.
- Lower bound for window-restricted estimators in the critical-window regime.
- Temporal validity vs changepoint evidence.
- Operational horizon regulation as an illustration, not the headline.

### Reframe
- Remove any suggestion that the cube-root law is universal.
- Treat ADWIN+UMR as a case study of horizon regulation, not a universal detector improvement.
- State clearly that the experiments validate the trade-off and the worst-case story, not a single law across all drift paths.

### Remove or downgrade
- Any universal "law of memory" wording.
- Claims that Gaussian drift experiments confirm cube-root as the general empirical law.
- Hurst-parameterized scaling from the main manuscript.

### Paper 1 identity
- "Useful memory under drift has a structural worst-case horizon."

## Paper 2: Regime-Dependent Useful-Memory Theory

### Core claim
- The optimal horizon exponent depends on temporal path geometry.
- Memory laws form a family, not a single universal exponent.

### Main objects
- Temporal path geometry.
- Hurst-parameterized scaling family.
- Regime-adaptive memory control.
- Resolvability / ceiling effects.

### Candidate theorem direction
- If the drift path has Hurst exponent $H$ and the carrier has statistical exponent $\beta$, then horizon scaling depends on both exponents.
- Target form:
  - $n^*(H,\zeta) \asymp \zeta^{-2/(1+2H)}$ for the $\beta=1/2$ carrier regime.
  - More generally, $n^*(\beta,H,\zeta)$ and $E_{\min}(\beta,H,\zeta)$ follow a two-exponent family.

### Missing pieces
- Matching lower bound for $H<1$.
- Online estimation of $H$ or a proxy robust enough for a controller.
- A principled adaptive regulator that uses regime estimates.

### Paper 2 identity
- "Useful memory has temporal path geometry."

## Bridge between the papers
- Paper 1: worst-case finite-memory tracking and structural floor.
- Paper 2: regime-dependent scaling laws and path-geometry adaptation.
- Paper 1 should end with a short, explicit future-work paragraph pointing to Paper 2.

## Recommended manuscript stance for v0.2.1
- Be conservative in the main paper.
- Keep the strongest proven claims.
- Leave the regime-dependent family as a clearly labeled outlook / next-paper direction.
