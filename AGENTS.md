You are a scientific research agent working on this project.

Your role is to help produce a rigorous scientific paper.

Never:
- claim certainty without evidence
- hide assumptions
- fabricate facts, citations, or results
- confuse hypotheses with conclusions
- write as if the paper were narrating its own development history
- write inventory prose that lists modules instead of expressing one scientific object
- organize the manuscript around inherited contrasts from past work
- center the paper on historical scaffolding such as the cube-root story, `minimum kernel`, or other project-internal labels

Always:
- reason step-by-step
- use scientific methodology
- quantify uncertainty when possible
- derive, test, simulate, calculate, and verify proactively
- search for disconfirming evidence and edge cases
- revise beliefs based on evidence
- continue investigation until meaningful progress is achieved
- write as if the paper were being created correctly from scratch today

Optimize for epistemic honesty, rigor, and conceptual unity.

The paper's identity:
- The paper is about the temporal validity of retained evidence under drift.
- The central statistical object is the temporal-validity horizon for finite-memory distribution tracking.
- The main law balances a finite-sample term and a drift-induced staleness term:
  `E d(\widehat P_t^{(n)}, P_t) <= C_K n^{-a} + C_S zeta n^H`.
- The induced optimal memory scale is `n^*(a,H) ~ (C_K / zeta)^{1/(a+H)}`.
- The paper is not a generalization of the cube-root law. The cube-root scaling is, at most, one member of the horizon family and should not organize the manuscript.

Manuscript architecture:
- Organize the paper by scientific function, not by project history.
- Preferred structure:
  - main object
  - general horizon law
  - drift regularity / staleness control
  - tractable proof model
  - structural lower bound
  - benchmark theorem(s)
  - operational extension if theorem-level or as an explicit conjecture
  - empirical signatures
  - limitations / open problems
- Distinguish claim status explicitly:
  - theorem
  - benchmark result
  - conjecture
  - open problem

Writing rules:
- Every paragraph should do one of only five things:
  - define the object
  - state a claim
  - explain a mechanism
  - present evidence
  - delimit scope
- Delete or rewrite paragraphs that explain the paper's own assembly, framing evolution, or comparison with past internal versions.
- Prefer declarative scientific prose over defensive or contrastive prose.
- Avoid repeated rhetorical structures such as:
  - `the paper now`
  - `the manuscript now`
  - `the point is not`
  - `not merely`
  - `not just`
  - `rather than`
  - `beyond the canonical`
  - `beyond the minimum kernel`
  - `what remains`
  - `the right question`
  - `this figure makes ... concrete`
- Use historical or contrastive remarks only when scientifically necessary, and then briefly.
- If a sentence mainly explains why a section exists, rewrite it so that it states the scientific content directly.

Terminology discipline:
- Prefer object-level terminology over project-internal labels.
- Prefer:
  - `finite-sample term` over `carrier term`
  - `finite-sample rate exponent` over `carrier exponent`
  - `temporal-validity horizon` or `finite-memory horizon` over `useful-memory horizon` when formal precision matters
  - `tractable one-dimensional proof model` over `minimum kernel`
  - `root-n rate regime` or the exact model name over `canonical regime`
  - `validity-detection lag` or `pre-detection validity loss` over `invalidity gap` / `detector-silent staleness` when scientific tone matters
  - `lower-bound construction` or `least-favorable profile` over excessive repeated use of `witness`
- Avoid `canonical/noncanonical` as a front-stage organizing contrast.

Scope discipline:
- A tractable proof model is an anchor, not the protagonist.
- Benchmark theorems and operational extensions support the same object; they are not separate papers inside the paper.
- Do not use `certified` as a substitute for theorem closure; if a statement is not proved, present it as a conjecture or open problem.
- Do not let open extensions dominate the exposition.

Compression discipline:
- Prefer a short, strong paper over a long inventory.
- Keep only material that serves the central object and its main claims.
- Push low-yield refinements, historical detours, and secondary constants out of the main line unless they are essential.

At the end of every turn, update the persistent project state inside `AGENTS.md` only within the dedicated delimited section below:

<!-- PROJECT_STATE_BEGIN -->
Project status:
- Core identity is settled: the paper is a theory of temporal validity of retained evidence under drift. The central object is the temporal-validity horizon for finite-memory distribution tracking, governed by `E d(\widehat P_t^{(n)},P_t) <= C_K n^{-a} + C_S zeta n^H` and `n^*(a,H) ~ (C_K/zeta)^{1/(a+H)}`.
- Main-line structure is stable: law -> drift/staleness control -> tractable proof model -> structural lower bound -> benchmark result -> operational extension or conjecture -> empirical signatures.
- Related-work anchors are settled enough for writing: empirical-Wasserstein finite-sample claims should cite `Fournier-Guillin`, `Weed-Bach`, and `Boissard-Le Gouic`; fixed-`epsilon` entropic geometry should cite `Genevay`, `Mena-Niles-Weed`, `del Barrio`, `Rigollet-Stromme`, `Stromme`, `Groppe-Hundrieser`, `Eckstein-Nutz`, `Ghosal-Nutz-Bernton`, and `Genans-Wintenberger`.
- Closed results: abstract upper law and optimized horizon law; exact uniform-window staleness constant; 1D bounded-support fixed-span root-`n` proof model; Gaussian lower-bound construction at exponent level; Gaussian location minimax benchmark; refined Gaussian ramp/endpoint constants.
- Operational extension status: fixed-`epsilon` Sinkhorn is theorem-level in the 1D compact-support class and remains conjectural in the embedded/high-dimensional fixed-span class.
- Numerical evidence now points to a moderate-band target `epsilon \in [0.2,0.8]`: `W_2(\widehat P_{tri},\widehat P_{iid})` slows in higher intrinsic dimension, while direct Sinkhorn band slopes stay near the parametric regime and Jacobian probes show the main obstruction lies in `xx/yy` self-couplings, not `xy`. TwoNN recovers the intrinsic class well after simple calibration and supports using estimated rather than imposed `k` in the synthetic frontier.
- The current proof target is a uniform derivative/equicontinuity theorem for the Sinkhorn derivative class on embedded fixed-span supports over a moderate compact band, preferably with moderate `epsilon_min` bounded away from zero.
- Main live blockers: prove the moderate-band embedded inheritance theorem or keep it explicitly conjectural; formalize the 1D bounded-support proof model on a natural support-preserving class if further theorem closure is desired; keep regular-family inheritance open unless an actual theorem is added.
- The worktree still contains a separate uncommitted two-column typography/layout experiment; it compiles cleanly, but it is not part of the settled manuscript state.

Closed results:
- Abstract upper law and optimized horizon law are closed.
- Uniform-window staleness is closed with exact finite-`n` constant `C_{H,n}` and asymptotic constant `(2H+1)^{-1/2}`.
- The tractable 1D bounded-support fixed-span triangular-array proof model yields the root-`n` finite-sample regime via quantile/Bahadur structure, and the conditional `W_2` derivation is now written out explicitly in the manuscript.
- Structural Gaussian lower-bound construction is closed at the exponent level for the root-`n` regime.
- Gaussian location is closed at minimax-rate level for deterministic Holder drift paths.
- Refined Gaussian lower-bound geometry exists: exact ramp frontier and endpoint-minimal profile `g_r^{min}=h^H-(h-r)^H` sharpen constants for `H<1`, and full derivations now live in the appendix `appendices/proof_details.tex`.

Broader theory:
- The correct extension principle is family- and geometry-specific horizon inheritance.
- If local metric geometry scales like `||theta-theta'||^alpha` under local regularity, then the natural metric finite-sample exponent is `a=alpha/2` and metric staleness exponent is `alpha H`.
- Distributional extension should target regular dominated families; support-changing/nonregular and singular atomic families are real obstructions.
- Gaussian scale benchmark exists in code; a full Gaussian-scale lower theorem remains open.

Operational extension:
- Fixed-`epsilon` Sinkhorn is an operational geometry, not a surrogate raw-`W_2` theorem.
- Closed structural ingredients: iid MID/LCA benchmark structure, exact support-complexity inheritance in the embedded fixed-span model, and dual threshold `alpha > k/2`.
- Pointwise and bandwise one-dimensional fixed-`epsilon` inheritance theorems are now written in the manuscript on the proof-model class: compact support + Proposition 2.4 + iid null debiased-Sinkhorn theory + algebraic transfer of Eckstein-Nutz value stability give `E S_epsilon(Phat_tri, Pbar) = O(n^{-1/2})` for each fixed `epsilon>0`, and the same constants are uniform on compact `epsilon`-bands.
- Current synthetic evidence (`alpha=2`, span `0.25`): maximal observed stable bands `0.5` for `(8,1)`, `(8,2)`, `(12,2)` and `0.2` for `(12,1)`; a denser `epsilon` sweep preserves the same band maxima while exposing a small mixed boundary region, and a compact-band fixed-support diagnostic script now reproduces the uniform-band 1D behavior directly.
- New embedded-band diagnostics sharpen the remaining operational blocker: `W_2(Phat_tri,Phat_iid)` has root-like slope only for `k=1` and slows around `0.44` for `k=2` and `0.28` for `k=4`, so the high-dimensional embedded theorem cannot be closed by the existing `W_2`-Lipschitz transfer. Direct debiased-Sinkhorn bootstrap slopes remain near the parametric regime across the tested `epsilon` grid, making a direct Sinkhorn differentiability/equicontinuity proof the plausible route.
- Direct `epsilon`-process diagnostics now support that route: sup-over-band debiased-Sinkhorn slopes are about `0.52` for triangular `(8,1)`, `0.51` for `(8,2)`, `0.43` for `(12,1)`, and `0.52` for `(12,2)`, with scaled log-`epsilon` modulus bounded across `n`. The remaining proof target is a uniform derivative/equicontinuity theorem for the Sinkhorn derivative class on embedded fixed-span supports.
- The full embedded-model triangular-array Sinkhorn horizon inheritance remains conjectural, not theorem-level; the remaining gap is the genuinely operational one: uniform derivative control over nontrivial `epsilon`-bands in the embedded/high-dimensional setting.
- Compact support plus `epsilon_min > 0` does give a standard uniform projective/Hilbert contraction for the Sinkhorn scaling iteration via bounded positive kernels, but no verified source here upgrades that to a uniform operator-norm bound for the fixed-point Jacobian or to uniform first/second Hadamard derivatives of the debiased Sinkhorn divergence.
- Numerical Jacobian probes now sharpen that gap: for the centered log-scaling map, the `xy` Jacobian stays comfortably contractive on bands with `epsilon_min >= 0.2`, and the `xx` inverse norm, while larger, does not grow with `n` there (for `k=2`, worst observed values are about `12` at `epsilon=0.2`, then settle near `5-7` by `n=64+`). The near-singular behavior concentrates at small `epsilon` (`0.05` and `0.1`) and in higher intrinsic dimension. This suggests the direct proof route should target moderate compact bands first, not bands approaching zero.
- A stronger moderate-band probe on `[0.2,0.8]` now supports that target for `k=2`: across `(8,2)` and `(12,2)`, the worst inverse norms still occur at small `n`, while the largest-tested `n` has mean inverse norms around `2.7` for all `xy/xx/yy` blocks. The practical proof target is therefore a theorem on a moderate compact band such as `[0.2,0.8]`, with the self-coupling Jacobians as the main technical burden.
- A larger-`n` follow-up (`n` up to `1024`) confirms the same pattern: in `[0.2,0.8]`, the `xx` and `yy` inverse norms plateau rather than drifting upward, with the worst values concentrated at small `n` and the large-`n` means settling near `2.6-4.8` depending on `epsilon` and intrinsic dimension. The current numerical hypothesis is that a uniform derivative/equicontinuity proof should be feasible first on this moderate band.
- A TwoNN calibration check on the synthetic embedded supports recovers the intrinsic class after a simple factor-2 calibration (`k_hat ≈ round(TwoNN/2)`) with high accuracy for `n ≥ 128`, and the moderate band `[0.2,0.8]` remains stable when the band diagnostics are grouped by the estimated class rather than the imposed one. That makes the moderate-band target robust to estimated, not imposed, intrinsic dimension in the current synthetic model.
- A new TwoNN operational diagnostic on the synthetic frontier gives a stronger, question-specific result on the supported 9-pair / 6-epsilon grid: `k_hat` beats `ambient_dim` on leave-one-out MAE for `epsilon_max` and on the pair-level stability cut at `epsilon=0.2`; the default CLI config now matches that supported setting, while the full rowwise stability map remains mixed.
- Scientifically, the TwoNN result says the operational frontier is keyed more to local intrinsic geometry than to ambient dimension in the tested embedded supports: `k_hat` is a better predictor of the stable-band cutoff, but the irregular rowwise pattern shows that intrinsic dimension alone still does not determine the whole frontier.
- The paper text has been scrubbed of project-facing metatext in the front matter, theory, Sinkhorn evidence, appendix diagnostics, conclusion, related-work, and limitations/open-problem sections: the prose now states object-level scientific content directly, with claim status expressed as theorem, conjecture, evidence, or open problem rather than as a roadmap or repository note.
- The open-problem ledger now reflects solved versus unsolved operational items more explicitly: the 1D compact-band fixed-`epsilon` question is no longer presented as open, while the embedded bandwise theorem is isolated as the operational open problem, with moderate-band self-coupling derivative control identified as the main technical gap.
- A narrower Gaussian bridge is now plausible as an intermediate theorem route: the linear-Gaussian entropic-OT / Riccati setting suggests a constructive streaming-Sinkhorn regime between the i.i.d. fixed-`epsilon` results and the fully general embedded triangular-array conjecture. This bridge is distinct from the main embedded open problem and should be treated as a candidate theorem family, not as a substitute for the general conjecture.
- A light editorial tightening pass has reduced excess in the abstract, introduction, empirical setup, related-work operational paragraph, limitations preamble, and conclusion without changing claim status. The remaining deliberate richness is mainly in the broad related-work span and the nine-item open-problem ledger; the validity-detection problem is secondary to the paper's central horizon object.
- The final phrasing keeps the citation density low in the empirical section by citing Facco et al. in Related Work and naming TwoNN there as the minimal-neighborhood proxy for local geometry; the empirical section then states the result directly as a regime-selection diagnostic.
- TwoNN should be mentioned only as a conceptual bridge from local intrinsic geometry to regime selection; avoid mechanistic explanation of the estimator inside the manuscript prose.
- The TwoNN / `GenansWintenberger2026` connection is now explicit at the level of calibration: intrinsic dimension is the relevant variable for selecting operational `\varepsilon`-bands, and TwoNN serves as the practical estimator for that local geometry in the empirical frontier section.

Empirical layer:
- The empirical core should stay lean: U-curve, roughness scaling, operational regime map, and validity loss before detector alarm.
- The main empirical phenomenon is that retained evidence can become invalid for the present before change becomes statistically detectable.
- Online policy experiments exist in code, but policy is secondary to the paper's central scientific object.

Main live blockers:
- Proof-completion blocker: formalize item (iv) of the 1D proof model for a natural bounded-support fixed-span class, ideally support-preserving smooth deformations or an equivalent verified class, so the current theorem-faithful empirical support becomes an actual theorem-level closure.
- Theorem-or-cut blocker: either prove the full bandwise fixed-`epsilon` embedded-model triangular-array inheritance on a nontrivial band/high-dimensional intrinsic class or keep that broader operational section explicitly subordinate and conjectural.
- Scope blocker: keep regular-family inheritance as an open conjectural extension unless an actual theorem beyond the 1D/Gaussian line is proved.
- Secondary but real theory gaps: sharp first-moment constant `C_K`, lower theory beyond the root-`n` regime, full Gaussian-scale lower theorem, and class-tight distributional lower bounds.
- Submission-strength extras, not core correctness: broader baselines, real-data validation, and system-level adaptive-policy work.

Proof-gap / refined constants:
- Old supplement claim `Gamma(a,H) >= 2` is false for the old support-based constant on `H in (0,1]`; fixed-`H` infimum is `2 min(1,R(H))`, with global infimum about `1.3200`.
- For refined Gaussian lower-bound constants (ramp and endpoint-minimal), `C_S^{asymp}(H)/C_H^{ref} > 1` on `H in (0,1]`, so the refined proof-gap infimum is exactly `2`, attained only as `a -> 0`.
- The analytic route for the refined claim is essentially identified via Komatsu-style Gaussian tail bounds, but the proof-gap result is no longer part of the principal main-text arc.
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
