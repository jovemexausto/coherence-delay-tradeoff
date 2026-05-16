# Bahadur-Kiefer Resolution for the i.n.i.d. Triangular Array

This note is the technical resolution path for the minimum kernel in `notes/main_theorem_package.md`.
It is not the paper-level theorem package; it explains why Proposition 3 is believable and where its boundary lies.

## Main point

The uniform remainder can be closed, but the correct rate is not just the Taylor rate.
The empirical increment term usually dominates and caps the remainder at the classical
`n^{-3/4}` scale (up to logs), just as in the i.i.d. case.

## Assumptions

Work on an interior quantile band `u \in [\epsilon, 1-\epsilon]`.

Assume:

- compact support;
- `\inf_{n,u} \bar f_n(\xi_n(u)) \ge c_1 > 0`;
- `\bar f_n` is uniformly Hölder on a neighborhood of the interior quantile range, i.e.
  `|\bar f_n(x)-\bar f_n(y)| \le L |x-y|^\alpha` for some `\alpha \in (0,1]`;
- the empirical CDF for intervals has the usual VC-type local modulus bound for independent
  non-identical Bernoulli indicators.

## Decomposition

Let `q_n(u)=\xi_n(u)` and `e_n(u)=\hat\xi_n(u)-\xi_n(u)`.

Then

`R_n(u) = A_n(u) + B_n(u)`

with

`A_n(u) = -[\Delta_n(q_n(u)+e_n(u)) - \Delta_n(q_n(u))]/\bar f_n(q_n(u))`,

`B_n(u) = -\rho_n(u)/\bar f_n(q_n(u))`,

where `\Delta_n = \mathbb F_n - \bar F_n` and

`\rho_n(u) = \bar F_n(q_n(u)+e_n(u)) - \bar F_n(q_n(u)) - \bar f_n(q_n(u))e_n(u)`.

## Rates

1. Since `e_n(u)=O_p(n^{-1/2})` uniformly on the interior, the interval length in `A_n`
   is `O_p(n^{-1/2})`.

2. The local empirical increment over an interval of length `h` is `O_p((h/n)^{1/2})`
   up to the usual logarithmic factor from uniformization. With `h = O_p(n^{-1/2})`,
   this gives

   `sup_u |A_n(u)| = O_p(n^{-3/4} polylog(n)).`

3. The Taylor remainder satisfies

   `|\rho_n(u)| \le L |e_n(u)|^{1+\alpha}`

   so

   `sup_u |B_n(u)| = O_p(n^{-(1+\alpha)/2}).`

Therefore

`sup_{u \in [\epsilon,1-\epsilon]} |R_n(u)| = O_p(n^{-\beta} polylog(n))`

with

`\beta = min(3/4, (1+\alpha)/2) > 1/2`.

In particular,

`sup_{u \in [\epsilon,1-\epsilon]} |R_n(u)| = o_p(n^{-1/2}).`

## Integrated `L_2`

Squaring and integrating gives

`\int_{\epsilon}^{1-\epsilon} E[R_n(u)^2] du = o(n^{-1})`.

The empirical increment term contributes `O(n^{-3/2})` after integration,
and the Taylor term contributes `O(n^{-(1+\alpha)})`.

## Correction to the earlier draft

The earlier claim `O_p(n^{-(1+\alpha)/2})` for the full remainder is too strong.
That rate only describes the Taylor part `B_n(u)`. The empirical increment term `A_n(u)`
is the classical bottleneck.

So the right statement is:

- `B_n(u)` improves with smoother densities;
- `A_n(u)` keeps the global rate near `n^{-3/4}`;
- both are still `o_p(n^{-1/2})`.

## Lab status

The current Python diagnostic confirms the split qualitatively:

- the empirical-increment contribution is much larger than the Taylor term;
- the reconstruction error is small relative to the full remainder;
- the full interior remainder still decays faster than `n^{-1/2}` in the tested fixed-span setup.

## Boundary sweep

The first frontier sweep suggests:

- varying `\epsilon` from `0.02` to `0.2` changes conditioning more than the rate;
- increasing the fixed span shifts the empirical rate slightly but does not break the `> 1/2` regime;
- the empirical term stays about two orders of magnitude larger than the Taylor term across the sweep;
- the reconstruction error remains tiny, so the decomposition is numerically stable.

This is consistent with the empirical increment term being the real bottleneck and with the fixed-span bounded-support theorem being robust.

## External literature synthesis

The external summary is compatible with this note and strengthens the minimum theorem target:

- triangular-array i.n.i.d. is the right formal setting;
- the interior assumptions are first-order only: uniform lower bound plus uniform Hölder continuity;
- the remainder can be stated at the classical `n^{-3/4}`-type scale up to logs, which is still enough for `o(n^{-1/2})`;
- the integrated remainder `o(n^{-1})` is the right quantity for the Wasserstein expansion.

So this does not change the direction of the research. It confirms that the minimum kernel can be stated with the existing assumptions, and that the remaining work belongs in the moderate and practically relevant layers.
