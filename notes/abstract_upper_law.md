# Abstract Upper Law

This note states the paper's main theorem object in a proof-ready form.
It is the top-level law that the carrier instantiations are meant to feed.

## Setup

Let `d` be any probability metric satisfying the triangle inequality.

At time `t`, let:

- `P_t` be the present target law;
- `\bar P_t^{(n)}` be the window target induced by the last `n` observations or weights;
- `\hat P_t^{(n)}` be the empirical or measurement-layer estimate built from that window.

The error decomposition is between:

- carrier error: `d(\hat P_t^{(n)}, \bar P_t^{(n)})`;
- staleness error: `d(\bar P_t^{(n)}, P_t)`.

## Theorem 1

Assume there exist constants `C_K > 0`, `C_S > 0`, exponent `a > 0`, roughness index `H in (0,1]`, and amplitude `zeta > 0` such that for all `n` in the regime of interest,

- `E d(\hat P_t^{(n)}, \bar P_t^{(n)}) <= C_K n^{-a}`;
- `d(\bar P_t^{(n)}, P_t) <= C_S zeta n^H`.

Then

`E d(\hat P_t^{(n)}, P_t) <= C_K n^{-a} + C_S zeta n^H`.

## Proof

By the triangle inequality,

`d(\hat P_t^{(n)}, P_t) <= d(\hat P_t^{(n)}, \bar P_t^{(n)}) + d(\bar P_t^{(n)}, P_t)`.

Take expectations and apply the two assumptions.

No further structure is needed at this level.

## Corollary 1

Let

`\Phi(n) = C_K n^{-a} + C_S zeta n^H`.

Balancing the two terms yields the useful-memory scale

`n^*(a,H,zeta) ~ (C_K / zeta)^{1 / (a + H)}`

up to constants depending only on `a`, `H`, and `C_S`.

At that scale,

`\Phi(n^*) ~ C_K^{H/(a+H)} zeta^{a/(a+H)}`

again up to constants depending only on `a`, `H`, and `C_S`.

## Proof sketch for the corollary

Set the two terms to the same order:

`C_K n^{-a} ~ C_S zeta n^H`.

This gives

`n^{a+H} ~ C_K / (C_S zeta)`.

Substituting the resulting scale back into either term gives the optimized error rate.

## Canonical slices

- `a = 1/2`, `H = 1` gives the cube-root scale `n^* ~ (C_K / zeta)^{2/3}`.
- `a = 1/2`, general `H` gives `n^* ~ (C_K / zeta)^{2/(1+2H)}`.
- changing the measurement layer changes `a` and `C_K`, not the logic of the theorem.

## What this theorem does and does not require

This theorem does require:

- a carrier bound for the chosen measurement layer;
- a staleness bound for the chosen path class.

This theorem does not require:

- one-dimensionality;
- `W_2` specifically;
- a particular estimator;
- a particular lower-bound witness.

That is why it is the right paper-level theorem object.
