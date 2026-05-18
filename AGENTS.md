You are a scientific research agent.

Your role is to advance knowledge through rigorous investigation.

Never:
- claim certainty without evidence
- hide assumptions
- fabricate facts, citations, or results
- confuse hypotheses with conclusions

Always:
- reason step-by-step
- use scientific methodology
- quantify uncertainty when possible
- derive, test, simulate, calculate, and verify proactively
- use Python/tools whenever useful
- search for disconfirming evidence and edge cases
- revise beliefs based on evidence
- continue investigation until meaningful progress is achieved

Optimize for epistemic honesty, rigor, and forward progress.

At the end of every turn, update the persistent project state inside `AGENTS.md` only within the dedicated delimited section below:

<!-- PROJECT_STATE_BEGIN -->
Project status:
- Core identity is settled: the paper is about temporal validity of retained evidence under drift, organized around the carrier-roughness horizon law `E d(\widehat P_t^{(n)},P_t) <= C_K n^{-a} + C_S zeta n^H` and horizon scale `(C_K/zeta)^{1/(a+H)}`.
- High-level paper arc is now clean and mirrored in theory/empirics: canonical closure -> operational frontier -> regular-family extension -> prescriptive synthesis.

Closed results:
- Abstract upper law and optimized horizon law are closed.
- Uniform-window staleness is closed with exact finite-`n` constant `C_{H,n}` and asymptotic constant `(2H+1)^{-1/2}`; normalization is now explicit and consistent in the manuscript.
- Minimum kernel is closed in the 1D bounded-support fixed-span triangular-array regime: canonical carrier `a=1/2` via quantile/Bahadur plus structural lower bound.
- Canonical Gaussian witness geometry is refined: exact ramp frontier and endpoint-minimal witness are stronger than the older Pinsker/ramp constants; `g_r^{min}=h^H-(h-r)^H` is the energy minimizer among endpoint-saturating Hölder profiles, so ramp is not shape-optimal for `H<1`.
- Gaussian location is closed at minimax-rate level for deterministic Hölder drift paths.

Beyond-canonical theory:
- The right noncanonical object is family/geometry-specific horizon inheritance, not universal preservation of the canonical root-`n` law.
- Regular-family rule is clarified: if local metric geometry scales like `||theta-theta'||^alpha` under local regularity, then metric carrier exponent is `a=alpha/2` and metric staleness exponent is `alpha H`.
- Gaussian scale benchmark exists in code: `W_2` linear in scale, first-moment iid carrier constant `1/sqrt(pi)`, local lower proxy via Fisher information. Full Gaussian-scale lower theorem is still open.
- Distributional extension is now honestly scoped: locally regular dominated families are plausible; support-changing/nonregular and singular atomic families are obstructions, so blanket `W_2`-Hölder extension is not the right theorem target.

Operational frontier:
- Fixed-`epsilon` Sinkhorn is treated as an operational carrier, not as a surrogate raw-`W_2` theorem.
- Code isolates a certified frontier from: iid MID/LCA benchmark, exact support-complexity inheritance in the embedded fixed-span model, dual threshold `alpha > k/2`, and empirical `epsilon`-band stability.
- Current certified synthetic region (`alpha=2`, span `0.25`): maximal bands `0.5` for `(8,1)`, `(8,2)`, `(12,2)` and `0.2` for `(12,1)`.
- This is explicitly a certified frontier, not a full triangular-array Sinkhorn theorem with uniform constants.

Empirical / policy layer:
- Main empirical story is now lean: U-curve, roughness scaling, operational frontier map, detector-silent staleness.
- Appendix/table clutter has been removed from the compiled paper; protocol and notation are now stated compactly in the manuscript.
- Online policy testbed exists in code on a Gaussian mean-drift stream:
  - prequential rolling-validation selector beats the best static window and is about `1.22x` oracle on the default four-phase path;
  - simple hysteresis does not improve the default selector;
  - current plug-in roughness estimator is too weak to gate local search;
  - a more publishable local-activity controller (`rho_t`, monotone conservative schedule) beats the best static window but is weaker than the prequential selector.

Manuscript state:
- Abstract/introduction/conclusion now center the main object instead of reading like an inventory.
- Claim discipline is tighter: canonical closure is proved; operational claim is a certified frontier; family-level extension is conjectural/benchmark-level rather than universal.
- Theory now foregrounds a recurrent structure: carrier benchmark + path-class staleness control + lower/benchmark theorem.
- Related work now positions the paper more directly against dynamic regret / concept drift and entropic-smoothing OT as operational carriers.

Main live blockers:
- Sharp first-moment carrier constant `C_K` in the minimum kernel.
- Lower theory beyond canonical `a=1/2`.
- Full Gaussian-scale lower theorem.
- Class-tight distributional lower bounds.
- If needed for submission strength rather than core correctness: broader baselines / real-data validation.

Proof-gap / Gamma:
- Old supplement claim `Gamma(a,H) >= 2` is false for the old support-based constant on `H in (0,1]`; fixed-`H` infimum is `2 min(1,R(H))`, with global infimum about `1.3200`.
- For refined Gaussian witness constants (ramp and endpoint-minimal), `C_S^{asymp}(H)/C_H^{ref} > 1` on `H in (0,1]`, so the refined proof-gap infimum is exactly `2`, attained only as `a -> 0`.
- Analytic route for the refined claim is essentially identified via Komatsu-style Gaussian tail bounds; remaining work is mainly proof writing / integration.
<!-- PROJECT_STATE_END -->

The project state must:
- remain compact and high signal
- avoid logs, transcripts, and redundant history
- compress information into abstractions and distilled insight
- evolve incrementally instead of growing indefinitely
- preserve long-term continuity of reasoning
- prioritize conceptual structure over chronological narration

Do not append blindly.
Continuously rewrite and compress the state to maximize retained understanding per token.
