# Research Program

## Current State

The project now separates into two distinct but connected lines of work.

The first line is the current manuscript: a conservative worst-case paper about
finite-memory tracking under drift. Its central object is the variance--staleness
trade-off under a local $W_2$-Lipschitz path bound. In that setting, the
finite-memory averaging class yields a cube-root horizon law, and the Gaussian
location lower bound shows that the induced floor is structural in the critical
window regime.

The second line is the broader research direction uncovered during the project:
the optimal horizon exponent may depend on the temporal geometry of the drift
path itself. That idea is scientifically stronger, but it is not yet mature
enough to subsume the current paper. It requires at least a clean path-class
formalism, matching lower bounds beyond the ramp-like worst case, and a serious
story for online regime estimation.

## Paper 1: What Is Settled

### Scientific identity
- Finite-memory tracking under drift has a structural worst-case floor.
- The cube-root horizon law is the correct minimax story for the linear-staleness
  regime induced by the local $W_2$-Lipschitz path bound.
- Temporal validity is distinct from changepoint evidence.

### Claims that are strong enough to keep
- Tracking error decomposes into a finite-sample term and a staleness term.
- The finite-memory averaging class yields a closed-form cube-root optimum.
- The Gaussian location construction provides a matching lower-bound story in the
  adversarial / ramp-like regime.
- UMR is a valid operationalization of the horizon law as a backend-agnostic
  temporal-validity cap.
- The cap-only regime is a real operational phenomenon: memory can already be
  stale while changepoint evidence is still insufficient.
- EMA-driven horizon control has a recoverability gap: contraction is easier than
  re-expansion.

### Claims that must stay out
- Any universal "law of memory" wording.
- Any implication that cube-root is the empirical law of every drift process.
- Any suggestion that ADWIN+UMR is the central contribution.
- Any Hurst/fBm/path-geometry theorem or notation.

### Proper role of UMR in Paper 1
- UMR is not a detector and not a detector booster.
- UMR is a shared horizon cap driven by a smoothed local drift proxy.
- It is backend-agnostic in concept and can be wrapped around detectors, but the
  detector wrapper is only one instantiation.

## Paper 2: What Exists and What Is Missing

### Core idea
- The horizon exponent may depend on temporal path geometry.
- Cube-root and square-root are likely members of a broader family.
- The Hurst exponent is a plausible path-class parameter for one important
  stochastic family, especially under fractional Brownian motion.

### Current theorem skeleton
- Deterministic Hölder-$H$ mean paths under Gaussian noise are the cleanest
  foundation.
- Uniform-window tracking over that class yields the bias--variance balance
  $C_K n^{-1/2} + c_H \zeta n^H$.
- Optimizing gives $n^*(H,\zeta) \asymp \zeta^{-2/(1+2H)}$ and
  $E_{\min}(H,\zeta) \asymp \zeta^{1/(1+2H)}$.
- A localized Le Cam bump construction should deliver the matching lower bound
  exponent.
- The cube-root and square-root laws then appear as the cases $H=1$ and
  $H=1/2$.

### Working note
- `PAPER2_HOLDER_THEOREM.md` contains the current theorem sketch and proof
  skeleton.

### What already looks novel
- Using a path roughness parameter to determine the exponent of the optimal
  memory horizon for distribution tracking.
- Unifying cube-root and square-root as members of a two-exponent scaling family.
- Interpreting path roughness as temporal geometry for memory policy.

### What is not ready yet
- The theorem needs a polished, fully written proof.
- The matching lower bound still needs a final constant-clean derivation.
- Online estimation of $H$ is a separate problem and currently unresolved.
- No implementation yet turns path-geometry estimation into a robust controller.

### Proper status
- Paper 2 is a real research direction.
- It is not ready to absorb Paper 1.
- It needs theorem work, estimator work, and a clean experimental design.

## UMR as a Software Trajectory

### Paper-1-safe form
- A temporal-validity cap.
- Input: local drift proxy.
- State: EMA-smoothed drift estimate.
- Output: horizon cap $n_t^*$.
- Usage: fixed windows, EWMA, multi-window ensembles, detector wrappers.

### Longer-term destination
- A native tracker, not just a cap.
- Regime inference.
- Adaptive horizon / forgetting policy.
- Multi-timescale state.
- Resolvability and uncertainty awareness.

That native tracker belongs to the broader research program and should not be
smuggled into the current paper.

## Bibliography Strategy

### Paper 1
- Keep the current transport / Sinkhorn / nonstationary optimization / concept
  drift / AoI references.
- Consider adding one direct adaptive-filtering reference if the related-work
  paragraph on EWMA, RLS, and Kalman stays broad.
- Consider adding explicit dataset provenance for Bikes if Bikes remains visible
  in the paper or appendix.
- Consider using `MenaWeed2019` if the Sinkhorn calibration discussion is
  sharpened.

### Paper 2
- Add the fBm and long-memory canon only when the paper exists as a separate
  object: Mandelbrot \\& Van Ness, Beran, Samorodnitsky \\& Taqqu, and related
  Hurst-estimation references.
- Keep these references out of Paper 1 unless a fully separate theorem or remark
  is actually introduced there.

## Benchmark Strategy

### For Paper 1
- Use synthetic geometry-heavy experiments to support the theorem.
- Keep real data as a restrained case study for temporal-validity gaps.
- Do not overpromise detector wins.

### For the broader program
- The strongest future targets are memory-sensitive prediction tasks rather than
  pure drift detection.
- Candidate areas: recommendation, online forecasting, multi-window ensembles,
  rolling classifiers, and other systems where stale memory is the actual
  bottleneck.

## Research Risks

### Risk 1
- The current empirical story can still over-index on ADWIN if the tables and
  captions are not carefully worded.

### Risk 2
- The current synthetic Gaussian sweep should not be described as universal
  confirmation of cube-root.

### Risk 3
- Paper 2 language can leak into Paper 1 through future-work phrasing or
  over-interpretation of new experiments.

### Risk 4
- UMR can still be misunderstood as a detector-specific hack unless the code,
  appendix, and empirical text stay aligned.

## Guiding Rule

Paper 1 should read as a complete, narrow, theorem-first paper.

Paper 2 should begin only after the missing lower-bound and regime-estimation
questions have become first-class objects rather than speculative bridges.
