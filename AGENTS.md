You are a scientific research agent working on this project.

Your role is to help produce a rigorous scientific paper.

Never:
- claim certainty without evidence
- hide assumptions
- fabricate facts, citations, or results
- confuse hypotheses with conclusions
- write as if the paper were narrating its own development history
- write inventory prose that lists modules instead of expressing one scientific object
- organize the manuscript around inherited contrasts such as `canonical/noncanonical`
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
- Core identity is settled: the paper is a theory of temporal validity of retained evidence under drift. Its central object is the temporal-validity horizon for finite-memory distribution tracking, governed by `E d(\widehat P_t^{(n)},P_t) <= C_K n^{-a} + C_S zeta n^H` and `n^*(a,H) ~ (C_K/zeta)^{1/(a+H)}`.
- Editorial reset is now mostly implemented: front matter, theory framing, empirical framing, conclusion, limitations, and appendix language read from the object outward rather than from project history. Main-line compression is stable: law -> proof model -> structural lower bound -> benchmark/operational regime -> empirical signatures.
- Related-work anchoring is settled enough for writing: raw empirical-Wasserstein finite-sample claims should run through `Fournier-Guillin (2015)`, `Weed-Bach (2019)`, and `Boissard-Le Gouic (2014)`; fixed-`epsilon` entropic geometry should run through `Genevay`, `Mena-Niles-Weed`, `del Barrio`, `Rigollet-Stromme`, `Stromme`, `Groppe-Hundrieser`, `Eckstein-Nutz`, `Ghosal-Nutz-Bernton`, and `Genans-Wintenberger (2026, arXiv:2602.04335)` when discussing geometry-aware calibration heuristics; projection-based alternatives should be framed as distinct proxy geometries, not automatic improvements for raw `W_2`.
- Lower-bound refinements and operational scope are now better subordinated in manuscript structure: refined Gaussian constants appear as one compact supporting proposition, and fixed-`epsilon` Sinkhorn is framed through structural closed pieces plus an explicit conjectural inheritance statement.
- A short-paper contract now exists in `notes/short-paper-contract.md`: it fixes the claim ledger, the paper/appendix/repository split, and a banned-from-main-text list. Auxiliary agents should treat that file as normative when deciding what belongs in the manuscript.
- The paper/appendix/repository split is now implemented without a dedicated guide section in the PDF: nonessential operational mechanics and family-level support figures have been removed from the main text, the appendix keeps only supporting diagnostics and proof details, and repository provenance stays outside the manuscript proper.
- The recent rigor pass closed two review-sensitive issues: the staleness formula is now written with the correct finite-`n` constant, and the 1D proof model plus refined Gaussian constants now have explicit derivation routes in the manuscript/appendix rather than floating as ungrounded sketches.
- The current worktree also contains an uncommitted two-column typography/layout test; it now compiles cleanly with `lmodern` and no remaining overfull equation warnings, but it is still a visual experiment rather than a settled manuscript state.
- The remaining prose cleanup is now essentially complete: conclusion, limitations, related work, and appendix language no longer speak in project-facing terms; the residual `minimum_kernel` and `carrier_layers` strings are internal labels in theorem/table artifacts, not front-facing narrative.
- External-review-sensitive framing is now sharper in the manuscript: the 1D proof model is explicitly presented as a narrow bounded-support fixed-span instantiation with an assumed integrated Bahadur remainder, and the fixed-`epsilon` Sinkhorn discussion now points directly to existing fixed-marginal differentiability/CLT and Gaussian entropic-OT results as i.i.d. anchors rather than leaving the operational conjecture bibliographically under-motivated.
- Front matter and empirical framing now state claim status more directly for readers: abstract/introduction distinguish closed results from the fixed-`epsilon` conjectural extension, and the empirical section now says explicitly that its role is structural signature evidence on synthetic streams rather than benchmark-leader evaluation or detector-uniform law discovery.
- New validation code now sharpens the frontier: support-preserving smooth fixed-support deformation diagnostics give Bahadur-residual exponents around `1.40-1.54` for integrated residual energy, so the 1D proof model now has theorem-faithful empirical support on a natural class; corresponding fixed-`epsilon` Sinkhorn slope checks remain bandwise and mixed, so they refine the conjecture's empirical scope but do not upgrade it.
- The fixed-`epsilon` literature check is now sharper: Goldfeld et al. provides iid debiased-Sinkhorn limit laws on compact support (`sqrt n` under the alternative, `n` under the null), del Barrio/Mena-type papers provide iid entropic-cost CLTs and fast bias bounds, and Eckstein-Nutz provides absolute-value stability for regularized OT values. That value bound transfers algebraically to debiased Sinkhorn on a common compact-support class with fixed quadratic cost, so the manuscript now has a pointwise fixed-`epsilon` 1D inheritance theorem on the proof-model class; what remains unavailable is a general uniform `epsilon`-band Lipschitz law or a high-dimensional embedded-model triangular-array theorem.

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
- A pointwise one-dimensional fixed-`epsilon` inheritance theorem is now written in the manuscript on the proof-model class: compact support + Proposition 2.4 + iid null debiased-Sinkhorn theory + algebraic transfer of Eckstein-Nutz value stability give `E S_epsilon(Phat_tri, Pbar) = O(n^{-1/2})` for each fixed `epsilon>0`.
- Current synthetic evidence (`alpha=2`, span `0.25`): maximal observed stable bands `0.5` for `(8,1)`, `(8,2)`, `(12,2)` and `0.2` for `(12,1)`; a denser `epsilon` sweep preserves the same band maxima while exposing a small mixed boundary region.
- The full embedded-model triangular-array Sinkhorn horizon inheritance remains conjectural, not theorem-level; the current faithful diagnostics support only bandwise stability, and the remaining gap is the genuinely operational one: uniform control over nontrivial `epsilon`-bands and higher-dimensional intrinsic classes, not the pointwise fixed-`epsilon` 1D case.

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
