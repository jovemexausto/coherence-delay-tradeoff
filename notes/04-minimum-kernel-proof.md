# 04. One-Dimensional Proof Details
Status: closed
Category: proof-model
Prev: 03. One-Dimensional Proof Model
Next: 05. Structural Lower Theory

This note records the proof structure for the 1D bounded-support fixed-span
model.

## Main steps

1. In one dimension,
   `W_2^2(\hat P_n^{tri}, \bar P_n) = \int_0^1 (\hat q_n(u)-q_n(u))^2 \, du`.

2. Use the Bahadur decomposition
   `\hat q_n(u)-q_n(u) = Z_n(u) + R_n(u)`
   with
   `Z_n(u) = (u-\hat F_n(q_n(u)))/\bar f_n(q_n(u))`.

3. Since the row is independent,
   `Var(\hat F_n(q_n(u))) <= 1/(4n)`.

4. A uniform positive lower bound on `\bar f_n` gives
   `\mathbb E\int_0^1 Z_n(u)^2 \, du = O(n^{-1})`.

5. By assumption,
   `\mathbb E\int_0^1 R_n(u)^2 \, du = o(n^{-1})`.

6. The cross term is `o(n^{-1})` by Cauchy--Schwarz.

7. Therefore
   `\mathbb E W_2^2(\hat P_n^{tri}, \bar P_n) = O(n^{-1})`,
   and Jensen yields the root-`n` rate.

## Status

This derivation now appears in the manuscript itself. The note remains only as
an internal proof summary.
