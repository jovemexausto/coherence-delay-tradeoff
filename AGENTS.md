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
- write object-first and final-form only. No draft-history prose, self-contrast, or section-purpose framing.

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
- Core objects: temporal-validity horizon for finite-memory distribution tracking under drift, with law `E d(\widehat P_t^{(n)},P_t) <= C_K n^{-a} + C_S zeta n^H`, asymptotic scale `n^*(a,H) ~ (C_K/zeta)^{1/(a+H)}`, exact continuous optimizer `n_star=((a C_K)/(H C_S zeta))^{1/(a+H)}`, and induced useful-memory region `U_delta={n:Phi(n)<= (1+delta)Phi(n_star)}`.
- Closed theory: abstract upper law; useful-memory region proposition with normalized profile `Psi(x)=(H x^{-a}+a x^H)/(a+H)` and linear scaling `U_delta=n_star I_delta(a,H)`; exact uniform-window staleness constant; abstract triangular-array Hadamard transfer lemma; a finite-dimensional smooth transfer proposition now gives a quadratic Taylor remainder criterion; one-dimensional root-`n` proof model under explicit Bahadur remainder assumptions; a smooth-deformation Bahadur corollary now makes the integrated remainder quantitative with `E∫R_n^2=O(n^{-2})` on the tested deformation class; a sliced/product multivariate corollary now transfers the horizon law under uniform projected control; Gaussian lower-bound exponent law; Gaussian location minimax benchmark; compact-support one-dimensional pointwise and bandwise fixed-`epsilon` Sinkhorn inheritance; a conditional embedded moderate-band Sinkhorn proposition now isolates the remaining self-coupling stability assumption; a weighted-memory proposition now shows uniform weights maximize effective sample size and linear ramps recover exact effective lag; Gaussian ramp/endpoint constants; smooth fixed-support and fixed-span triangular-array Bahadur diagnostics now quantify lower-order residual decay and span dependence on the tested classes.
- Empirical layer: synthetic signatures for U-curve, roughness scaling, and validity-detection lag; real-stream ELEC2 diagnostic on the `nswprice` marginal showing an interior retention optimum and useful-memory band; operational Sinkhorn evidence concentrated on moderate compact regularization bands, with intrinsic-geometry diagnostics serving as regime proxies where needed; online `twonn_geometry` diagnostics showing reliable recovery of `k` and strong improvement of aggregated-lag `H` estimation over endpoint slopes in moderate roughness regimes, with the online figure caption now stripped of redundant section wording; a theorem-ready-band report and table now give positive maximal bands on the calibrated Sinkhorn grid; a positive self-coupling stability proxy now holds on the calibrated `(8,2)` and `(12,2)` moderate-band pairs; a smooth multivariate transfer proxy now shows an interior U-curve and lower-order linearization residual under triangular-array drift; a minimal tapered-weight sensitivity check on the default synthetic stream now reports slightly lower mean error for triangular/geometric tapers than the uniform window, alongside positive lag-weight `W1` from uniform; theorem-style Python tests now cover non-uniform weights via exact effective-sample-size and effective-lag identities on linear ramps.
- Online controller status: `online_horizon_adaptation.py` now uses the aggregated roughness-scale estimate itself in the structural controller while keeping the activity proxy for band selection; on the default phase profile it decisively improves on the naive plug-in rule and beats both the pure activity baseline and the best static window, though the wider validation-based adaptive hybrid remains stronger.
- Current manuscript identity: object-first theory of temporal validity under drift, with a coherent sequential arc now ordered as theory, empirical signatures, implications for online adaptation, related work, limitations, and conclusion; related-work framing is rewritten around mutual contribution rather than topic contrast, now including an explicit tracking-versus-DRO distinction relative to `KeehanAndersonWiesemann2025`, and appendix language mirrors the same object-first tone. Verifiable 1D sufficient conditions now sit in their own subsection, the multivariate sliced/product extension now follows the lower-bound/benchmark line, empirical subsection labels and captions are normalized for consistent navigation, the bibliography has been pruned to cited/useful entries only, and the limitations section now organizes five open problems so that Problems 1--2 close the current theorem line, Problem 3 isolates memory design, and Problems 4--5 define the subsequent online theory program.
- Proof-route anchors from the current literature scan: multivariate empirical transport splits into raw Wasserstein dimension/intrinsic-dimension routes, smooth-Wasserstein delta-method routes, and fixed-`epsilon` entropic routes (MID/LCA on compact `epsilon` bands); the most realistic Sinkhorn triangular-array theorem path runs through Hadamard differentiability plus a stable Jacobian/influence operator on a moderate `epsilon` band; the strongest bibliography gaps are now the local-stationarity bridge (`Dahlhaus`, `Vogt`, `Kurisu`, and the 2026 Wasserstein-local-stationarity paper of `Tinio--Alaya--Bouzebda`), theorem-bearing sliced-statistics papers already in the bib but underused (`NadjahiEtAl2020`, `NietertEtAl2022`), and weighted-empirical/Wasserstein precursors for non-uniform memory (`Mason2016` plus the older weighted empirical-process line); streaming OT computation remains secondary, with `GenevayEtAl2016` mainly useful for low-priority operational positioning.
- Next-cycle plan: first close embedded fixed-`epsilon` Sinkhorn on a moderate band via direct linearization, uniform self-coupling stability, and a triangular-array delta-method transfer; then pursue a multivariate transfer lemma for smooth/Hadamard-differentiable functionals, with a toy moment-vector-plus-smooth-map check as the executable proxy for the Jacobian/remainder structure; in parallel harden Bahadur sufficient conditions, add theorem-oriented Python tests for non-uniform weights via `n_eff` and lag-moment summaries, and keep online adaptation at plug-in containment level. Validation-detection theory remains last unless a detector-specific theorem appears cheaply.
- Verification status: core theorem and weighting tests currently pass (`26` Python tests in the expanded verification sweep), the weighted-memory identities and the new effective-sample-size proposition match their closed forms on linear ramps, the certified-band report is positive on the calibrated Sinkhorn grid, the self-coupling certificate is positive on the calibrated moderate-band pairs, the multivariate transfer diagnostic has an interior U-curve with a lower-order linearization residual, and the current Sinkhorn certificate remains positive on the calibrated synthetic grid. This turn's repository snapshot is staged for commit and archival.
- Open blockers: embedded bandwise Sinkhorn inheritance beyond the compact-support one-dimensional closure; sharp lower theory beyond the current benchmark family and root-$n$ regime; verified bounded-support Bahadur remainder classes; online useful-memory adaptation with guarantees; validity-detection theory. The tapered-weight diagnostic and theorem-style tests now isolate a credible non-uniform-memory law candidate via `n_eff` and lag moments, but whether it supports a robust theorem or sharper constants remains open.
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
