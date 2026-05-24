# 13. Sinkhorn Self-Coupling Lemma
Status: active
Category: theorem target
Prev: 12. Theorem Ladder
Next: -

## Target statement

Fix the calibrated embedded fixed-span model with intrinsic dimension `k=2` and
moderate band `B=[eps_min, eps_max]` bounded away from zero.

The target lemma is a uniform centered self-coupling stability statement of the form

1. the centered scaling fixed-point map is differentiable on `B`;
2. the centered self-coupling derivative blocks are uniformly invertible on `B`;
3. the inverse operator norm is uniformly bounded on `B`;
4. the resulting linearization remainder is `o(n^{-1/2})` uniformly on `B`.

The calibrated proxy evidence currently available is positive on `(8,2)` and
`(12,2)`, where both self blocks keep positive spectral gap, worst spectral radius
stays below `0.97`, and largest-sample mean inverse norm stays below `3.0`.

## Proof route

1. Rewrite the Sinkhorn scaling equations in centered coordinates.
2. Express the derivative blocks in a centered Schur-complement form.
3. Show the centered self-coupling block is a strict contraction on `B`.
4. Deduce uniform invertibility and a uniform inverse bound.
5. Propagate the bound through the implicit-function linearization.
6. Verify the remainder is lower order than `n^{-1/2}` on the embedded class.

## Failure modes

- the self-coupling block loses strict contraction at the lower edge of the band;
- the inverse norm inflates with ambient dimension even at fixed intrinsic dimension;
- the centered remainder requires a support-separation assumption that is not
  available uniformly;
- the proof only closes pointwise in `epsilon` and not uniformly on the band.
