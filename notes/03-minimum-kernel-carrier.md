# 03. Minimum-Kernel Carrier
Status: closed
Category: carrier
Prev: 02. General Law
Next: 04. Minimum-Kernel Proof

Minimum-kernel carrier in theorem-ready form.
Its role is to close the first rigorous carrier instantiation for the abstract upper law.

The consolidated proof narrative is in `notes/04-minimum-kernel-proof.md`.

## Setting

Let `X_{n,1}, ..., X_{n,n}` be independent one-dimensional observations with laws `P_{n,1}, ..., P_{n,n}`.

Define the window-mixture target

`\bar P_n = (1/n) \sum_{j=1}^n P_{n,j}`

and let `\hat P_n^{tri}` be the empirical measure of the triangular array sample.

Let `\bar F_n` be the CDF of `\bar P_n`, let `q_n = \bar F_n^{-1}`, and let `\hat q_n` be the empirical quantile function of `\hat P_n^{tri}`.

## Safe-zone assumptions

Assume:

- bounded support: there exists a fixed interval `[L,U]` containing the support of every `P_{n,j}`;
- fixed span: the within-window drift span is uniformly bounded and does not grow with `n`;
- absolute continuity on the interior band of interest;
- interior lower density bound: for some `\varepsilon in (0,1/2)`, `\inf_{n,u in [\varepsilon,1-\varepsilon]} \bar f_n(q_n(u)) >= c_0 > 0`;
- interior Hölder regularity: `|\bar f_n(x) - \bar f_n(y)| <= L |x-y|^\alpha` on a neighborhood of the interior quantile range for some `\alpha in (0,1]`;
- a uniform triangular-array Bahadur representation on `u in [\varepsilon,1-\varepsilon]` with integrated remainder `\int_{\varepsilon}^{1-\varepsilon} E[R_n(u)^2] du = o(n^{-1})`.

## Proposition 3

Under the safe-zone assumptions,

`E W_2^2(\hat P_n^{tri}, \bar P_n) = O(n^{-1})`

and therefore

`E W_2(\hat P_n^{tri}, \bar P_n) = O(n^{-1/2})`.

This is the canonical `a = 1/2` carrier instantiation used by the paper.

## Proof route

1. In one dimension,
   `W_2^2(\hat P_n^{tri}, \bar P_n) = \int_0^1 (\hat q_n(u) - q_n(u))^2 du`.

2. On the interior band, use the Bahadur representation
   `\hat q_n(u) - q_n(u) = (u - \hat F_n(q_n(u))) / \bar f_n(q_n(u)) + R_n(u)`.

3. The leading term has variance `O(1/n)` after integration because the array is independent and the fixed-span design only changes constants.

4. The integrated remainder is `o(n^{-1})` by assumption, with the needed decomposition and rate discussion consolidated in `notes/04-minimum-kernel-proof.md`.

5. The boundary band contributes only lower-order mass under bounded support and the interior density assumptions.

6. Summing the interior leading term, the lower-order cross term, and the remainder gives `E W_2^2 = O(n^{-1})`.

7. Jensen gives `E W_2 <= (E W_2^2)^{1/2} = O(n^{-1/2})`.

## Design effect interpretation

The exponent is the same as the i.i.d. mixture benchmark.
The triangular design changes the leading constant through a design effect, not through a change of exponent.

That comparison is summarized directly in `notes/04-minimum-kernel-proof.md`.

## What remains to write cleanly

The remaining consolidation task is to merge four ingredients into a single proof writeup:

- the quantile representation;
- the triangular-array variance calculation;
- the Bahadur-Kiefer remainder bound;
- the boundary-band cleanup.

These ingredients are now assembled in `notes/04-minimum-kernel-proof.md`.
The remaining work is to port that proof into the manuscript in final paper style.
