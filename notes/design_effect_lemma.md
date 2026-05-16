# Design Effect Lemma

This note is a local ingredient for Proposition 3 in `notes/main_theorem_package.md`.
It is not a paper-level theorem by itself; it explains why the minimum kernel should differ from the i.i.d. benchmark only through constants.

## Setting

Let `P_1, ..., P_n` be independent laws on a common interval `[L, U]` in one dimension, and let

`\bar P_n = (1/n) \sum_{j=1}^n P_j`.

Let `\hat P_n^{tri}` be the empirical measure of the fixed-design triangular array `X_j ~ P_j`, and let `\hat P_n^{iid}` be an i.i.d. sample from `\bar P_n`.

Assume:

- bounded support `supp(P_j) \subset [L, U]` for all `j`,
- densities `f_j` exist and are uniformly bounded above and below on the interior quantile band of interest,
- the mixture density `\bar f` is likewise bounded away from zero on that band,
- a uniform Bahadur-type quantile representation holds with `o(n^{-1/2})` remainder after integration over `u \in [\varepsilon, 1-\varepsilon]`.

## Lemma (design effect, fixed span)

Under the assumptions above, the `W_2` carrier is root-`n` for both sampling schemes:

`E W_2(\hat P_n^{tri}, \bar P_n) = C_tri n^{-1/2} + o(n^{-1/2})`

and

`E W_2(\hat P_n^{iid}, \bar P_n) = C_iid n^{-1/2} + o(n^{-1/2})`.

Moreover, the leading constants satisfy the pointwise design-effect inequality

`C_tri <= C_iid`

whenever the quantile-process variance is computed from the fixed design versus the mixture benchmark.

## Proof sketch

1. In one dimension,
   `W_2^2(\hat P_n, \bar P_n) = \int_0^1 (\hat q_n(u) - q_n(u))^2 du`,
   where `q_n = \bar F_n^{-1}` is the mixture quantile function.

2. Under a uniform Bahadur representation,
   `\hat q_n(u) - q_n(u) = (u - \hat F_n(q_n(u)))/\bar f(q_n(u)) + R_n(u)`,
   with `\int E[R_n(u)^2] du = o(1/n)`.

3. For the triangular array,
   `Var(\hat F_n(x)) = n^{-2} \sum_j F_j(x)(1-F_j(x))`.
   For the i.i.d. mixture,
   `Var(\hat F_n^{iid}(x)) = n^{-1} \bar F_n(x)(1-\bar F_n(x))`.

4. Since `x(1-x)` is concave,
   `n^{-1}\sum_j F_j(x)(1-F_j(x)) <= \bar F_n(x)(1-\bar F_n(x))` pointwise in `x`.

5. Plugging this into the Bahadur formula yields the constant inequality `C_tri <= C_iid` after integration over the quantile band.

## Interpretation

This is a pure design effect:

- same exponent (`1/2`),
- potentially different constants,
- triangular design removes random mixture-count fluctuations.

The lemma is the right local target for proving the minimum kernel and for identifying the constant-level design effect inside the abstract carrier law.
