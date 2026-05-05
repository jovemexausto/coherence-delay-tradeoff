# Review Response Report

## Purpose

This report is the shared reference for all follow-up phases. It records what the review got right, what must be corrected in the paper, what evidence already exists, and what still needs new proofs, ablations, or baselines.

## Executive Decision

We will pursue the strongest defensible version of the contribution, even if that requires new proofs and new ablations. The paper must reflect evidence, not defend a claim after the evidence weakens it.

The project remains centered on two flagship ideas:

1. finite-memory tracking under drift has a real coherence-delay floor;
2. intervention-aware diagnostics can expose coercive masking in action-coupled systems.

## Review Questions and Answers

### 1. How general is the cube-root law?

Current answer: the clean derivation is rigorous for the finite-memory averaging class already written in the paper, with sliding windows and EWMA as the verified cases. The current text is too broad if it suggests a universal minimax statement over all adaptive estimators.

What we will do:
- either prove a formal minimax lower bound for an explicit class of finite-memory kernels, or
- narrow the theorem language to a precise estimator class if the stronger proof does not close.

Preferred route:
- prove a lower bound for a class of causal, finite-memory averaging kernels with effective size and effective lag parameters.

### 2. Is there a sign error in Proposition 6.12?

Current answer: the live source currently states the positive cube-root scaling, but we must audit the manuscript and any derived text for stale `\zeta^{-1/3}` remnants.

What we will do:
- audit every occurrence of the proposition and surrounding discussion,
- verify the PDF and appendix text,
- add an explicit errata note if any stale version survived in a figure caption or response artifact.

### 3. How far does the ISS template go?

Current answer: the Borromean stability argument is a conditional template, constructively verified in the Gaussian case. It is not yet an operational theorem for arbitrary black-box `\Phi`.

What we will do:
- keep the template framing honest,
- strengthen the Gaussian instantiation if possible,
- explicitly separate theorem, template, and future work in the paper.

### 4. Are the action-argument necessity claims too strong?

Current answer: yes, the “must depend on three arguments” framing is stronger than necessary and risks sounding more novel than it is.

What we will do:
- map the discussion more directly to standard belief-state / POMDP / filtering language,
- keep the three-role decomposition as a modeling choice, not a universal ontological claim.

### 5. How robust is the effort-aware score?

Current answer: the score is promising, but its behavior under proxy mis-specification, `\lambda`, and `E0` is not yet fully pinned down.

What we will do:
- expand sensitivity and ablation analyses,
- treat `\sigma_A` as exact only in deterministic or constructively specified settings, and as a surrogate elsewhere,
- add explicit proxy-quality criteria.

Current ablation evidence:
- the particle and KuaiRand `\lambda` sweeps already show the stable operating range,
- the domain-specific `E0` choices are documented in the paper,
- exact / identified / proxy semantics are separated in the paper and appendix.

Operational ladder:
- `exact`: deterministic action channels or constructively specified replicas.
- `identified`: settings where the action channel is directly logged well enough to estimate the target quantity with a tractable estimator.
- `proxy`: logged or observational settings where the target is only approximated by an operational surrogate.

### 6. How honest is the KuaiRand interpretation?

Current answer: it is useful, but off-policy confounding and exposure bias remain real limitations.

What we will do:
- tighten the language around logged-data proxies,
- add stronger controls and matched comparisons,
- avoid causal claims that the data cannot support.

Current operational stance:
- KuaiRand is a logged-data surrogate benchmark, not a causal identification study.
- The reported score is useful because it tracks exposure concentration and masking, not because it equals the unobserved conditional entropy exactly.
- `\sigma_A` in KuaiRand belongs to the proxy level of the ladder.

### 7. Are there enough baselines?

Current answer: not yet.

What we will do:
- add CUSUM,
- add forgetting-factor RLS,
- add scalar Kalman predictors,
- add OT / representation-space baselines (Wasserstein or Fréchet / MMD-style),
- benchmark them under the same calibration discipline.

### 8. What about Sinkhorn runtime and practical cost?

Current answer: the paper says enough to justify the geometry, but not enough to justify deployment cost.

What we will do:
- measure throughput and latency,
- record dimension / window / epsilon trade-offs,
- distinguish theoretical rate from deployed runtime.

### 9. Should we rename the diagnostics and the project?

Current answer: yes, but only after the review-response work is stabilized.

Preferred naming direction:
- `C` -> `CI` (`Coherence Index`)
- `C^E` -> `EAC` (`Effort-Adjusted Coherence`)
- `M_t` -> `Masking Gap`
- umbrella: `CoherenceGuard`

We will not force a rename if it weakens the paper or obscures the math.

## Evidence Already Available

- sliding-window cube-root slope is already near `1/3`
- EWMA also shows the same scaling
- particle masking clearly separates apparent coherence from effort-aware coherence
- KuaiRand shows the logged/active regime where effort-aware diagnostics matter
- passive benchmarks still favor generic drift detectors like ADWIN

## Main Gaps That Still Need Work

- formal lower bound for the finite-memory class
- `\sigma_A` operational validity and proxy analysis
- additional baselines and fair calibration
- runtime / latency of the Sinkhorn pipeline
- stronger controls for KuaiRand / logged data

## Implementation Rule

Every future phase must update the paper if the evidence changes the claim.
