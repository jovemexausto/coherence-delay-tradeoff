# 01. Main Theorem Package
Status: active
Category: map
Prev: -
Next: 02. General Law

Claim structure.

## Central object

Temporal-validity horizon for finite-memory distribution tracking under drift.

The main law is

- `E d(\hat P_t^{(n)}, P_t) <= C_K n^{-a} + C_S zeta n^H`.

The induced horizon scale is

- `n^*(a,H) ~ (C_K/zeta)^{1/(a+H)}`.

## Closed theorem line

### Theorem: abstract upper law

- finite-sample term relative to the window target;
- staleness term relative to the present target;
- triangle-inequality decomposition.

### Corollary: optimal temporal-validity horizon

- `n^*(a,H) ~ (C_K/zeta)^{1/(a+H)}`;
- optimal risk scale `~ C_K^{H/(a+H)} zeta^{a/(a+H)}`.

### Lemma: exact finite-`n` uniform-window staleness constant

- `W_2^2(\bar P_t^{(n)},P_t) <= (zeta^2/n) sum_{j=0}^{n-1} j^{2H}`;
- `W_2(\bar P_t^{(n)},P_t) <= C_{H,n} zeta n^H`;
- `C_{H,n} -> (2H+1)^(-1/2)`.

### Proposition: tractable 1D proof model

- conditional root-`n` finite-sample rate in the bounded-support fixed-span triangular array;
- manuscript proof now written explicitly;
- Bahadur input remains an explicit hypothesis of the proof model.

### Proposition: structural Gaussian lower bound

- exponent-level lower law in the root-`n` regime;
- horizon is structural, not an estimator artifact.

### Theorem: Gaussian location minimax benchmark

- class-level benchmark on deterministic Holder paths;
- matching exponent-level upper and lower laws.

## Supporting benchmark result

### Proposition: Gaussian lower-bound constants

- exact ramp constant;
- endpoint-minimal profile;
- strict constant improvement for `H < 1`.

This is a supporting benchmark result, not a second theorem line.

## Conjectures

- fixed-`epsilon` Sinkhorn horizon inheritance;
- regular-family horizon inheritance.

## Open problems

- sharp first-moment constant `C_K` in the 1D proof model;
- lower theory beyond the root-`n` regime;
- full Gaussian-scale lower theorem;
- class-tight distributional lower bounds.
