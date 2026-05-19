# 02. General Law
Status: closed
Category: law
Prev: 01. Main Theorem Package
Next: 03. One-Dimensional Proof Model

Top-level law in proof-ready form.

## Setup

Let `d` be a probability metric satisfying the triangle inequality.

At time `t`, let:

- `P_t` be the present target law;
- `\bar P_t^{(n)}` be the window target induced by the last `n` observations or weights;
- `\hat P_t^{(n)}` be the estimator built from that window.

The decomposition is between:

- finite-sample error: `d(\hat P_t^{(n)}, \bar P_t^{(n)})`;
- staleness error: `d(\bar P_t^{(n)}, P_t)`.

## Theorem

Assume constants `C_K > 0`, `C_S > 0`, exponent `a > 0`, roughness index
`H in (0,1]`, and amplitude `zeta > 0` such that

- `E d(\hat P_t^{(n)}, \bar P_t^{(n)}) <= C_K n^{-a}`;
- `d(\bar P_t^{(n)}, P_t) <= C_S zeta n^H`.

Then

- `E d(\hat P_t^{(n)}, P_t) <= C_K n^{-a} + C_S zeta n^H`.

Proof: triangle inequality, then expectation.

## Optimized horizon

Let

- `\Phi(n) = C_K n^{-a} + C_S zeta n^H`.

Balancing the two terms yields

- `n^*(a,H) ~ (C_K/zeta)^{1/(a+H)}`
- `\Phi(n^*) ~ C_K^{H/(a+H)} zeta^{a/(a+H)}`

up to constants depending only on `a`, `H`, and `C_S`.

## Interpretation

This theorem does not depend on a particular geometry, estimator, or lower-bound
construction. It is the paper-level law into which proof models, benchmark
theorems, and conjectural extensions feed their own finite-sample exponents.
