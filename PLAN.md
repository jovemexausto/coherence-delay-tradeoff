# PLAN

This roadmap is driven by `REVIEW_RESPONSE.md`. Every phase below must update
the paper, code, figures, and report together whenever the evidence changes.

## Working Rule

- Keep exactly one task in `in_progress` at any time.
- When a phase is done, mark it `completed` and commit that phase.
- Do not start the next phase until the current phase is committed.
- Use `REVIEW_RESPONSE.md` as the standing source of truth for review answers,
  evidence gaps, and claim boundaries.
- Never defend a claim once the evidence points the other way.

## Evidence Rule

- Passive streams: keep ADWIN/KSWIN/Page-Hinkley as legitimate baselines and do not overclaim superiority.
- Active or logged systems: optimize for intervention-aware diagnosis, where the renamed diagnostics (`CI`, `EAC`, `Masking Gap`) should become the flagship layer if they survive the evidence.
- New names, visuals, or architecture only stay if they improve clarity, reproducibility, or measured performance.
- Product claims must be backed by benchmark tables, artifacts, and manuscript text in the same phase.

## Completed Program: Paper Revision

- [x] Phase 1: Formal Theory and EWMA
- [x] Phase 2: Coercive Masking as a Flagship Contribution
- [x] Phase 3: Calibration and Sensitivity
- [x] Phase 4: Scope and Honesty of the Stability Template
- [x] Phase 5: Related Work and Editorial Framing
- [x] Phase 6: Real-World Benchmarks and Limitations
- [x] Phase 7: Passive-vs-Active Framing
- [x] Phase 8: Editorial Reframing
- [x] Phase 9: Final Validation and Cleanup

## Program 2: Review Response and Strengthening

## Phase 10: Review Response Report and Claim Audit

- [x] Write `REVIEW_RESPONSE.md` as the shared reference for all later phases.
- [x] Audit the live manuscript against the review and record any stale claims.
- [x] Identify which review criticisms are already resolved, which are partially resolved, and which still require new evidence.
- [x] Mark the sign inconsistency / Proposition 6.12 issue as resolved in the live source, then verify no stale text remains.
- [completed] Commit Phase 10.

Why this phase exists:
- The review must become an execution spec, not just a rebuttal note.
- Every later phase should be traceable to a question, a gap, or an evidence-backed decision in `REVIEW_RESPONSE.md`.

## Phase 11: Theoretical Strengthening and Lower Bounds

- [x] Tighten the cube-root discussion so it is precise about what is proven, what is conjectured, and what is operationally supported by experiments.
- [x] Prove a formal lower bound for an explicit class of finite-memory estimators / kernels.
- [x] Add the strongest possible relation to nonstationary optimization, dynamic regret, and adaptive filtering without overclaiming equivalence.
- [x] Update the manuscript and the report together with the final theorem statement.
- [completed] Commit Phase 11.

Why this phase exists:
- This is the biggest remaining theoretical vulnerability, and the review is explicit about it.

## Phase 12: Effort Proxy, `\sigma_A`, and KuaiRand Honesty

- [x] Separate exact, synthetic, and logged-data interpretations of `\sigma_A` in the paper.
- [x] Formalize the observability ladder for `\sigma_A` (`exact`, `identified`, `proxy`) and make it explicit in the paper.
- [x] Add an explicit operational hierarchy for effort proxies and state when each one is valid.
- [x] Tighten the KuaiRand wording so it clearly reflects logged-data limitations, confounding risk, and the meaning of the reported metric.
- [x] Add any needed ablations on `\lambda`, `E0`, and proxy choice, and remove any unsupported causal language.
- [in_progress] Commit Phase 12.

Why this phase exists:
- The reviewer’s critique of effort estimation is real and must be answered with both language and evidence.

## Phase 13: Baseline Expansion and Fair Comparisons

- [ ] Implement CUSUM, forgetting-factor RLS, and scalar Kalman baselines on the real-world streams.
- [ ] Add at least one representation-space OT / Fréchet / MMD-style baseline if it can be implemented faithfully.
- [ ] Re-evaluate ELEC2 and Bikes under the same matching rules used in the paper.
- [ ] Re-evaluate active/logged benchmarks with calibration-matched comparisons and record any losses honestly.
- [ ] Commit Phase 13.

Why this phase exists:
- The paper becomes stronger if it can win in the right regimes and lose cleanly where it should.

## Phase 14: Computational Budget and Sinkhorn Cost

- [ ] Measure runtime / latency / throughput for the estimator and transport machinery under the streaming settings used in the paper.
- [ ] Separate theoretical sample-complexity claims from practical wall-clock behavior.
- [ ] Measure the impact of `\varepsilon`, window size, and dimension on cost and bias.
- [ ] Update the manuscript and report with the concrete computational picture.
- [ ] Commit Phase 14.

Why this phase exists:
- If the geometry is part of the contribution, the computational cost must be part of the evidence.

## Phase 15: Naming and Visual System

- [ ] Decide whether the renamed diagnostics should become the default (`CI`, `EAC`, `Masking Gap`) or whether a different naming stack is better.
- [ ] Decide the project umbrella name, with `CoherenceGuard` as the current leading candidate.
- [ ] Design flagship visualizations that communicate regimes, not just calibration tables.
- [ ] Update the paper figures and repo-facing docs if the new naming and visuals survive the evidence.
- [ ] Commit Phase 15.

Why this phase exists:
- The current notation is mathematically compact but product-poor.

## Phase 16: Software Core and Benchmark Harness

- [ ] Factor the repository into a clearer production-style core: scoring, detectors, calibration, adapters, evaluation, and visualization.
- [ ] Add stable APIs and typed outputs for the main diagnostics and alerts.
- [ ] Keep the experiment harness aligned with the core rather than duplicating logic.
- [ ] Preserve reproducibility artifacts as first-class outputs.
- [ ] Commit Phase 16.

Why this phase exists:
- The project should become a software system, not only a paper with scripts.

## Phase 17: Paper Refresh and Release Candidate

- [ ] Refresh the manuscript so it reflects the best surviving names, strongest proofs, new baselines, and new computational evidence.
- [ ] Remove or narrow any claim that later phases weaken.
- [ ] Rebuild the PDF and verify the release state against `REVIEW_RESPONSE.md`.
- [ ] Commit Phase 17.

Why this phase exists:
- The paper must remain the truthful public face of the current system.

## Notes

- The target is not universal dominance on passive drift detection.
- The target is a system that is clearly stronger in intervention-aware monitoring and materially better overall through evidence-backed strengthening.
- The preferred sequence is: review response report -> theory strengthening -> effort proxy honesty -> baselines -> runtime -> naming/visuals -> software core -> paper refresh.
