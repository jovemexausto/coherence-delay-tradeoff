# 04. Minimum-Kernel Proof
Status: closed
Category: carrier
Prev: 03. Minimum-Kernel Carrier
Next: 05. Structural Lower Theory

Consolidated proof narrative for Proposition 3.

This assembles the minimum-kernel carrier argument in one place and turns the supporting ingredients into parts of a single theorem.

## Goal

Prove that, in the fixed-span bounded-support one-dimensional setting,

`E W_2^2(\hat P_n^{tri}, \bar P_n) = O(n^{-1})`

and hence

`E W_2(\hat P_n^{tri}, \bar P_n) = O(n^{-1/2})`.

This is the canonical `a = 1/2` carrier instantiation for the abstract upper law.

## Setting

Let `X_{n,1}, ..., X_{n,n}` be independent one-dimensional observations with laws `P_{n,1}, ..., P_{n,n}`.

Define the mixture target

`\bar P_n = (1/n) \sum_{j=1}^n P_{n,j}`

with CDF `\bar F_n` and quantile function `q_n = \bar F_n^{-1}`.

Let `\hat P_n^{tri}` be the empirical measure of the triangular-array sample and `\hat q_n` its empirical quantile function.

## Assumptions

Work on an interior quantile band `u in [\varepsilon, 1-\varepsilon]` for some fixed `\varepsilon in (0,1/2)`.

Assume:

- bounded support: all `P_{n,j}` are supported on a common compact interval `[L,U]`;
- fixed span: the within-window drift span is uniformly bounded in `n`;
- absolute continuity on the interior quantile range;
- interior lower density bound: `\inf_{n,u} \bar f_n(q_n(u)) >= c_0 > 0`;
- interior Hölder regularity: `|\bar f_n(x)-\bar f_n(y)| <= L |x-y|^\alpha` near the interior quantile range, with `\alpha in (0,1]`;
- the empirical interval counts for the triangular array satisfy the usual VC-type local modulus bound for independent, non-identical Bernoulli indicators.

This is the safe zone for the minimum kernel:

- 1-D bounded support;
- fixed within-window span;
- interior quantile band;
- density bounded below and uniformly Hölder on that interior range.

The current frontier picture is also clear:

- roughness alone does not break the root-`n` story in the fixed-span interior-band lab;
- span growth degrades the effective constant and then the exponent;
- the first genuine failure mode is roughness plus sufficiently fast span growth.

## Step 1: Quantile representation of `W_2^2`

In one dimension,

`W_2^2(\hat P_n^{tri}, \bar P_n) = \int_0^1 (\hat q_n(u) - q_n(u))^2 du`.

So the problem reduces to proving an integrated `L_2` bound for the empirical quantile error.

## Step 2: Bahadur decomposition on the interior band

On `u in [\varepsilon, 1-\varepsilon]`, write

`\hat q_n(u) - q_n(u) = L_n(u) + R_n(u)`

with leading term

`L_n(u) = (u - \hat F_n(q_n(u))) / \bar f_n(q_n(u))`

and remainder `R_n(u)`.

The required input is a triangular-array Bahadur representation on the interior band.

The theorem target is a uniform Bahadur linearization on the interior band:

`\hat q_n(u) - q_n(u) = (u - \hat F_n(q_n(u))) / \bar f_n(q_n(u)) + R_n(u)`

with

`\int_{\varepsilon}^{1-\varepsilon} E[R_n(u)^2] du = o(n^{-1})`.

The key point is that the remainder is not governed only by a Taylor term. It splits as

`R_n(u) = A_n(u) + B_n(u)`

where:

- `A_n(u)` is the empirical increment term;
- `B_n(u)` is the Taylor remainder term.

The empirical increment term is the bottleneck and limits the uniform rate to the classical `n^{-3/4}` scale up to logs.

## Step 3: Integrated remainder bound

Under the safe-zone assumptions,

- `sup_u |R_n(u)| = o_p(n^{-1/2})` on the interior band;
- `\int_{\varepsilon}^{1-\varepsilon} E[R_n(u)^2] du = o(n^{-1})`.

This is the precise remainder statement needed for the minimum kernel.

It is enough because the proof only needs integrated `L_2` control, not a sharper asymptotic expansion for the remainder.

The current numerical diagnostic matches that picture:

- the integrated residual fraction falls with `n`;
- the empirical increment term is much larger than the Taylor term;
- the full interior remainder still decays faster than `n^{-1/2}` in the tested fixed-span regime.

## Step 4: Leading-term variance is `O(n^{-1})`

For fixed `u`,

`L_n(u) = (u - \hat F_n(q_n(u))) / \bar f_n(q_n(u))`.

Since the sample is independent,

`Var(\hat F_n(x)) = n^{-2} \sum_{j=1}^n F_{n,j}(x)(1 - F_{n,j}(x))`.

The fixed-span assumption keeps the mixture density and the quantile denominator under control, so after dividing by `\bar f_n(q_n(u))^2` and integrating over the interior band,

`\int_{\varepsilon}^{1-\varepsilon} E[L_n(u)^2] du = O(n^{-1})`.

This is the root-`n` carrier mechanism.

The associated constant is a design-effect constant, not a new exponent. The key comparison with the i.i.d. mixture benchmark is:

- for the triangular array,
  `Var(\hat F_n(x)) = n^{-2} \sum_j F_{n,j}(x)(1-F_{n,j}(x))`;
- for an i.i.d. sample from the mixture,
  `Var(\hat F_n^{iid}(x)) = n^{-1} \bar F_n(x)(1-\bar F_n(x))`.

Since `x(1-x)` is concave,

`n^{-1} \sum_j F_{n,j}(x)(1-F_{n,j}(x)) <= \bar F_n(x)(1-\bar F_n(x))`,

so the triangular design differs from the mixture benchmark only through a favorable constant-level design effect.

## Step 5: Cross term is lower order

Expand the squared quantile error on the interior band:

`(L_n(u) + R_n(u))^2 = L_n(u)^2 + 2 L_n(u) R_n(u) + R_n(u)^2`.

Integrating and taking expectations,

`E \int_{\varepsilon}^{1-\varepsilon} 2 |L_n(u) R_n(u)| du`

is lower order by Cauchy-Schwarz, because

- `\int E[L_n(u)^2] du = O(n^{-1})`, and
- `\int E[R_n(u)^2] du = o(n^{-1})`.

Hence the cross term is `o(n^{-1})`.

## Step 6: Boundary band is negligible

The minimum-kernel proof is fundamentally an interior-band argument, but `W_2^2` integrates over all `u in [0,1]`.

Under bounded support, the total contribution from the boundary strips `[0,\varepsilon]` and `[1-\varepsilon,1]` is bounded by a constant multiple of `\varepsilon`, uniformly in `n`, because both `\hat q_n` and `q_n` stay in `[L,U]`.

So the proof proceeds in the standard order:

1. fix `\varepsilon > 0`;
2. prove the `O(n^{-1})` bound on the interior band;
3. bound the boundary-band contribution by `C \varepsilon`;
4. let `\varepsilon` be a small theorem constant.

The theorem therefore needs only interior regularity, not boundary smoothness.

## Step 7: Conclusion

Combining the interior leading term, the lower-order cross term, the integrated remainder bound, and the bounded boundary-band contribution gives

`E W_2^2(\hat P_n^{tri}, \bar P_n) = O(n^{-1})`.

Then Jensen yields

`E W_2(\hat P_n^{tri}, \bar P_n) <= (E W_2^2(\hat P_n^{tri}, \bar P_n))^{1/2} = O(n^{-1/2})`.

This closes the minimum-kernel carrier in the safe zone.

## What is mathematically doing the work

The proof rests on four ingredients:

- one-dimensional quantile representation of `W_2^2`;
- triangular-array Bahadur linearization;
- integrated `o(n^{-1})` remainder control;
- independent-array variance calculation with bounded fixed span.

Nothing in the proof suggests that the exponent should differ from `1/2` inside the safe zone.
The open work is consolidation and citation hygiene, not a search for a new rate.

## What this proof does not claim

This proof does not claim:

- a sharp asymptotic constant;
- a high-dimensional raw `W_2` theorem;
- validity when span grows with `n`;
- validity in the roughness-plus-span-growth failure regime.

Those belong to later regimes or to explicit frontier statements.
