# 03. One-Dimensional Proof Model
Status: closed
Category: proof-model
Prev: 02. General Law
Next: 04. One-Dimensional Proof Details

Tractable 1D bounded-support fixed-span triangular-array model closing the
first rigorous finite-sample regime.

## Setting

Let `X_{n,1}, ..., X_{n,n}` be independent one-dimensional observations with
laws `P_{n,1}, ..., P_{n,n}` and define the window target

- `\bar P_n = (1/n) sum_{j=1}^n P_{n,j}`.

Let `q_n` be the quantile function of `\bar P_n` and `\hat q_n` the empirical
quantile function of the triangular-array empirical measure.

## Proposition

Under:

- common compact support;
- fixed within-window span;
- mixture density bounded below and uniformly Holder;
- a Bahadur representation with integrated remainder `o(n^{-1})`;

one has

- `E W_2^2(\hat P_n^{tri}, \bar P_n) = O(n^{-1})`;
- `E W_2(\hat P_n^{tri}, \bar P_n) = O(n^{-1/2})`.

## Status

The conditional derivation is explicit. The only nontrivial hypothesis is the
integrated triangular-array Bahadur remainder.

## Interpretation

This is the proof-model anchor for the root-`n` finite-sample regime.
