# 01. Main Theorem Package
Status: active
Category: map
Prev: -
Next: 02. General Law

Current map of the paper's theoretical package after the recent constant-level
refinements. It separates:

- the closed paper core;
- sharp refinements that strengthen the canonical lower regime without changing the underlying theory;
- frontier questions that remain open.

The primary object is unchanged: the temporal validity of retained evidence for
distribution tracking under drift.

## 1. Closed paper core

### Theorem 1: Abstract upper law

Assume:

- `E d(\hat P_t^{(n)}, \bar P_t^{(n)}) <= C_K n^{-a}`;
- `d(\bar P_t^{(n)}, P_t) <= C_S zeta n^H`.

Then

- `E d(\hat P_t^{(n)}, P_t) <= C_K n^{-a} + C_S zeta n^H`.

This remains the top-level theorem object. It is the right paper-level law
because it separates the carrier side from the path-class staleness.

Reference: `notes/02-general-law.md`.

### Corollary 1: Optimized useful-memory horizon law

Balancing the two terms gives

- `n^*(a,H) ~ (C_K / zeta)^{1/(a+H)}`;
- `R^*(a,H,zeta) ~ C_K^{H/(a+H)} zeta^{a/(a+H)}`.

Special cases:

- `a = 1/2`, `H = 1`: cube-root horizon;
- `a = 1/2`, general `H`: the roughness-indexed family `2/(1+2H)`.

Reference: `notes/02-general-law.md`.

### Lemma 2: Uniform-window Hölder staleness

For a path in `P_H(zeta)` and a uniform window,

- `W_2^2(\bar P_t^{(n)}, P_t) <= (zeta^2 / n) sum_{j=0}^{n-1} j^{2H}`.

So the exact finite-`n` constant is

- `C_{S,n}(H) = (n^{-(2H+1)} sum_{j=0}^{n-1} j^{2H})^{1/2}`

and therefore

- `W_2(\bar P_t^{(n)}, P_t) <= C_{S,n}(H) zeta n^H`.

As `n -> infinity`,

- `C_{S,n}(H) -> (2H+1)^(-1/2)`.

This is now stronger than the earlier informal `c_H` notation because it makes
the window-level constant explicit.

For uniform windows, the natural asymptotic candidate sharp constant is
`(2H+1)^(-1/2)`. This should be treated as a refined upper-envelope constant, not
as a completed sharp theorem for the full family.

### Proposition 3: Minimum-kernel carrier

In the bounded-support fixed-span one-dimensional triangular array,

- `E W_2^2(\hat P_n^{tri}, \bar P_n) = O(n^{-1})`;
- hence `E W_2(\hat P_n^{tri}, \bar P_n) = O(n^{-1/2})`.

This is the first rigorous carrier instantiation feeding the abstract upper law.

References:

- `notes/03-minimum-kernel-carrier.md`;
- `notes/04-minimum-kernel-proof.md`.

### Proposition 4: Structural lower bound on the canonical `a = 1/2` regime

On the Gaussian location witness subclass,

- `Risk >= c_H sigma^{2H/(2H+1)} zeta^{1/(2H+1)}`

for some `c_H > 0`, so the useful-memory scale is structural on the canonical
`a = 1/2` regime.

This is the lower-bound fact the paper needs most: the horizon is not just the
optimum of one procedure.

Reference: `notes/05-structural-lower-theory.md`.

## 2. Sharp refinements already discovered

These refinements strengthen the canonical lower regime. They do not change the
underlying theory, but they matter for the internal state of the theory.

### 2.1 Exact Gaussian ramp frontier

If the ramp witness

- `mu_{t-j}^{\pm,h} = \pm beta (h-j)_+^H`

is paired with the exact Gaussian two-point testing error, then the large-ratio
problem reduces to a scalar root `x_H` solving

- `p_H Phi(-x_H) = x_H phi(x_H)`, where `p_H = 2H / (2H+1)`.

This gives the asymptotic constant

- `C_H^{ramp} = (2H+1)^{H/(2H+1)} x_H^{2H/(2H+1)} Phi(-x_H)`

and the horizon shape parameter

- `A_H^{ramp} = (sqrt(2H+1) x_H)^{2/(2H+1)}`.

This strictly improves the current Pinsker-style witness constant on the same
subclass.

Reference: `notes/06-exact-gaussian-witness-frontier.md`.

### 2.2 Witness-shape extremality

Inside the larger class of endpoint-saturating discrete `H`-Hölder profiles, the
ramp is not extremal when `H < 1`.

The pointwise minimal feasible profile is

- `g_r^{min} = h^H - (h-r)^H`.

It minimizes the testing energy among all endpoint-saturating profiles and
coincides with the ramp only at `H = 1`.

Reference: `notes/07-witness-shape-extremality.md`.

### 2.3 Stronger lower-regime constant via the endpoint-minimal witness

The endpoint-minimal witness has energy constant

- `I_H = 2H^2 / ((H+1)(2H+1))`.

So the exact Gaussian lower-regime constant improves further to

- `C_H^{min} = I_H^{-H/(2H+1)} x_H^{2H/(2H+1)} Phi(-x_H)`.

This keeps the same exponent and same structural horizon law while strictly
strengthening the subclass lower constant for `H < 1`.

References:

- `notes/06-exact-gaussian-witness-frontier.md`;
- `notes/07-witness-shape-extremality.md`.

## 3. What belongs in the main manuscript

The principal manuscript should definitely contain:

- the abstract upper law;
- the optimized useful-memory horizon law;
- the exact finite-`n` uniform-window staleness lemma;
- the minimum kernel;
- the structural lower bound on the canonical `a = 1/2` regime;
- the extended and operational regimes with explicit status labels.

### Extended regime

The extended regime is the intended theorem form for low-dimensional or
low-intrinsic-dimensional carrier inheritance beyond the minimum kernel.

Reference: `notes/08-extended-regime.md`.

### Operational regime

The operational regime is the intended theorem form for measurement geometries,
such as fixed-`epsilon` Sinkhorn, that appear stable enough to operationalize the
general law beyond fragile raw-`W_2` settings.

Reference: `notes/09-operational-regime.md`.

The manuscript may also contain, if space permits:

- a short remark on the exact Gaussian ramp frontier;
- a short remark that the ramp witness is not shape-optimal for `H < 1`.

Those are refinements of the canonical lower regime, not replacements for the main theorem
package.

## 4. What remains open

Four open directions now stand out.

1. Promote the exact Gaussian ramp frontier from a numerically verified note to a
fully written asymptotic proposition.

2. Decide whether the endpoint-minimal witness already gives the best subclass
lower bound in a wider Hölder witness class, or whether an even stronger shape
exists once one relaxes endpoint normalization or uses richer witness families.

3. Extend the lower theory beyond the canonical `a = 1/2` regime by building
carrier-matched witnesses for other exponents.

4. Close the extended-regime and operational-regime carrier theorems beyond the
minimum kernel.

## 5. Sharp-constant cautions

The sharp-constant frontier is scientifically useful, but it should not be
allowed to rewrite the status of the theorem package.

- do not claim a completed sharp theory for the full `(a,H)` family;
- do not claim a sharp first-moment carrier constant such as `C_K = 1/sqrt(6)`
  without a separate first-moment argument;
- do treat the exact finite-`n` staleness constant and the lower-regime witness
  refinements as real advances inside the already-closed theory.

## 6. Current status line

The theory is now best summarized as follows.

- The exponents are closed at paper level.
- The staleness constant for uniform windows is explicit.
- The minimum kernel is the first rigorous carrier.
- The horizon is structurally real on the canonical `a = 1/2` regime.
- The lower-regime constants are now materially better understood than before.
- The remaining frontier is no longer vague: it is a sharp list of carrier,
  witness-shape, and beyond-canonical-regime questions.
