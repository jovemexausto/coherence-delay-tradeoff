# PAPER 2

## Working Identity

Primary working title:
`Useful Memory Has Temporal Path Geometry`

Alternative technical titles:
- `A Holder-Parameterized Horizon Law for Finite-Memory Tracking`
- `Temporal Path Geometry of Useful Memory Under Drift`
- `Regime-Dependent Horizon Laws for Endpoint Tracking Under Temporal Roughness`

Recommended one-sentence thesis:
The exponent governing the useful-memory horizon is not universal; it depends on temporal path geometry, and for deterministic Holder-H drift classes the optimal memory and minimax risk obey a regime-dependent law.

90-second reviewer message:
Paper 1 establishes the worst-case Lipschitz envelope and its cube-root horizon. Paper 2 explains that this is only one member of a larger family. For deterministic Holder-H path classes, the tracking error decomposes into a statistical term and a staleness term growing like `zeta n^H`, leading to `n^*(H,zeta) ~ zeta^{-2/(1+2H)}` and minimax risk `R^*(H,zeta) ~ zeta^{1/(1+2H)}` up to constants. The upper and lower exponents match. The remaining open problem is not the scaling law, but the sharp minimax constants and their optimal-recovery interpretation.

## Status Summary

### What is already known

The conceptual structure of Paper 2 is real, not speculative.

Closed at the exponent level:
- deterministic Holder-H drift class is the correct mathematical object;
- upper bound has the form `C_K n^{-1/2} + c_H zeta n^H`;
- lower bound via bump construction plus Le Cam matches the exponent;
- minimax risk exponent is `1/(1+2H)`;
- optimal memory exponent is `-2/(1+2H)`;
- `H=1` recovers the cube-root law;
- `H=1/2` recovers the square-root law.

Not closed at the constant level:
- theorem-proof upper constants are not sharp;
- current lower bound constant is not sharp;
- the real open problem is the minimax constant, not the exponent.

### What Paper 2 is responsible for

Paper 2 must do four things well:
- formalize the Holder-H path class cleanly;
- prove the regime-dependent exponents cleanly;
- explain how Paper 1 appears as the `H=1` worst-case member;
- state honestly what is not yet solved, especially sharp constants.

It does not need to solve sharp constants to be a strong paper.

## Non-Negotiable Positioning

### Paper 2 is not

- a direct theorem about fractional Brownian motion;
- a paper about ADWIN or detector wrapping;
- a vague narrative about `different regimes` without theorem-level closure;
- a paper that requires sharp constants to exist.

### Paper 2 is

- a deterministic temporal-regularity theorem;
- a regime family for useful-memory horizons;
- the conceptual unification of cube-root and square-root as special cases;
- a bridge from worst-case memory validity to temporal path geometry.

## Main Theoretical Message

For deterministic Holder-H drift classes, the variance-staleness trade-off becomes

```tex
\mathbb E\,\mathcal E(n)
\le C_K n^{-1/2} + c_H \zeta n^H,
```

and balancing the two terms yields

```tex
n^*(H,\zeta) \asymp (C_K/\zeta)^{2/(1+2H)},
```

with minimax risk scaling

```tex
R^*(H,\zeta)
\asymp
\sigma^{2H/(1+2H)}\zeta^{1/(1+2H)}.
```

The core discovery is not only that exponents vary with `H`, but that useful memory has temporal path geometry.

## Mathematical Object

### Recommended primary class

Paper 2 should be stated for a deterministic Holder-H path class.

Representative form:

```tex
\mathcal H(H,\zeta)
=
\left\{ (P_t^*)_t : d(P_t^*,P_s^*) \le \zeta |t-s|^H \text{ for all relevant } s,t \right\}
```

where `d` is the endpoint geometry being tracked in the theorem statement.

### Recommended interpretation

This class is an envelope model for temporal roughness.

That means:
- the theorem is deterministic;
- the theorem does not depend on stochastic-process construction;
- Hurst language can be heuristic or motivational only.

### What not to claim

Do not claim:
`fractional Brownian motion with Hurst exponent H lies in the class almost surely with fixed deterministic zeta`.

Correct statement:
paths from stochastic families with Hurst-type roughness can motivate the deterministic Holder envelope, but the theorem itself is not yet an fBm theorem.

## Main Theorem Package

Paper 2 should revolve around a clean theorem package.

### Theorem A: Upper bound for uniform-window tracking

Need:
- deterministic Holder-H path assumption;
- explicit finite-sample term;
- explicit staleness term scaling as `zeta n^H`;
- clean constants;
- explicit role of `sigma` or `C_K`.

Representative target form:

```tex
\mathcal E(n)
\le
C_K n^{-1/2} + c_H \zeta n^H.
```

### Corollary B: Optimized horizon

Need the optimizer and minimized risk written symmetrically and cleanly.

Representative target form:

```tex
n^*(H,\zeta)
\asymp
\left(\frac{C_K}{\zeta}\right)^{2/(1+2H)},
\qquad
\mathcal E_{\min}(H,\zeta)
\asymp
C_K^{2H/(1+2H)}\zeta^{1/(1+2H)}.
```

If `sigma` is separated from `C_K`, do it explicitly and consistently.

### Theorem C: Lower bound via hard pair / bump construction

Need:
- a lower bound matched in exponent;
- precise scope on the estimator class;
- explicit statement of the critical-window regime or optimized-window problem;
- honest non-sharp constant language.

Representative target form:

```tex
R^*(H,\zeta)
\ge
C_{\mathrm{LB}}(H)
\sigma^{2H/(1+2H)}\zeta^{1/(1+2H)}.
```

### Corollaries D and E: Special cases

These should be explicit and visible.

- `H=1` gives cube-root.
- `H=1/2` gives square-root.

This is an important editorial payoff because it shows Paper 1 is a special case, not a contradiction.

## What Must Be Corrected from the Current Theorem Block

### 1. Rename the theorem away from `Hurst-parameterized`

Current issue:
the theorem is about deterministic Holder classes, not fBm itself.

Recommended replacement:
- `Holder-Parameterized Horizon Law`
- or `Temporal-Roughness-Parameterized Horizon Law`

### 2. Fix the fBm remark

Current issue:
too strong and mathematically inaccurate.

Required replacement idea:
`The deterministic Holder class can be viewed as an envelope model for stochastic path families with Hurst-type roughness, but the theorem itself is deterministic and does not rely on an fBm construction.`

### 3. Make lower-bound scope precise

Current issue:
some formulations sound too broad over all estimators using the last `n` observations while the actual argument lives at the critical scale or optimized-window level.

Recommended fix:
choose one of the following and state it explicitly.

Option A:
the lower bound is for the optimized-window minimax problem.

Option B:
the lower bound is for windows in the critical scale regime.

Recommendation:
Option A is editorially cleaner if it can be stated cleanly.

### 4. Factor `sigma` symmetrically in upper and lower statements

Required final scaling form:

```tex
R^*(H,\zeta)
\asymp
\sigma^{2H/(1+2H)}\zeta^{1/(1+2H)}.
```

The paper should not have `sigma` hidden in one side and absent in the other.

### 5. Rewrite the tightness remark

Correct final message:
- exponents are tight;
- theorem-proof upper constants are non-sharp;
- Le Cam lower bound seems to capture roughly half of the best observed linear constant;
- closing the constant gap likely needs tools beyond two-point Le Cam.

### 6. Remove the `H -> 0` corollary if it overclaims

If the current wording suggests independence from `zeta`, remove it.

It is not needed for the paper's main argument and creates unnecessary attack surface.

## What Is Closed vs Open

### Closed enough for Paper 2 main claim

- deterministic Holder-H class framing;
- upper theorem;
- lower theorem with matching exponent;
- regime law `1/(1+2H)` and `2/(1+2H)`;
- `H=1` and `H=1/2` special cases;
- interpretation of Paper 1 as worst-case member.

### Still open

- sharp minimax constants;
- exact extremal path and extremal estimator characterization;
- whether the optimal estimator is uniform, nearly uniform, or a nontrivial kernel;
- full stochastic transfer to fBm-style models;
- online estimation of `H` or regime class;
- geometry-adaptive controller design.

## Sharp Constants: Final Strategic Position

### Bottom line

Sharp constants are not a must-have for Paper 2.

Paper 2 remains strong without them if it delivers:
- exponent-tight theory;
- a clean deterministic Holder framing;
- convincing special cases and interpretation;
- honest discussion of the constant gap.

### How to describe the open problem in the paper

Recommended wording:
`We establish the sharp scaling exponents, but not the sharp minimax constants. Closing the remaining constant gap appears to require tools beyond two-point Le Cam, likely through an optimal-recovery or modulus-of-continuity analysis tailored to endpoint estimation under temporal regularity constraints.`

### What the sharp-constant problem really is

It is not just `improving a constant`.

It is about identifying:
- the true extremal path geometry;
- the true optimal estimator or kernel;
- the exact information-theoretic obstruction behind endpoint estimation;
- the right dual or modulus formulation of the problem.

### Is it a separate paper?

Yes, very plausibly.

Natural future-paper identities:
- `Sharp Minimax Constants for Endpoint Estimation under Temporal Holder Drift`
- `Optimal Recovery for Temporal Endpoint Estimation`
- `Sharp Constants in Holder-Parameterized Memory Horizon Laws`

### What solving sharp constants would change

Scientific impact:
- closes the theory;
- identifies the actual extremal problem behind the scaling law;
- clarifies whether current upper procedures are nearly optimal or structurally suboptimal.

Practical impact:
- improves fine calibration of horizon rules;
- clarifies how much uniform windows lose versus the true optimum;
- sharpens policy comparisons.

But the main gain is theoretical, not practical.

## Numerical State of the Constants Problem

Current validated pattern from numerical investigation:
- theorem upper constants are loose;
- the exact uniform-window risk is much better than the theorem upper bound;
- the best observed nonnegative linear kernel improves only a few percent over uniform;
- the Le Cam lower bound appears stably around half of the best observed linear constant.

Representative pattern already observed:
- `C_LB / C_best-linear ~ 0.53` across tested `H` values.

Interpretation:
- exponents look genuinely closed;
- constants are not closed;
- the problem is now a sharp-constant lower-bound problem more than an upper-bound crisis.

Important caution:
- no strong claim should be made yet about signed weights or full optimality beyond the observed linear searches.

## Literature Positioning

The best strategic positioning for the sharp-constant discussion is not `Pinsker solved this already`.

More credible framing:
- Le Cam gets the exponent and a non-sharp constant;
- Assouad is useful for rates but not obviously the constant solution here;
- van Trees may help locally but is unlikely to fully close the endpoint constant;
- the most promising path for sharp constants looks like optimal recovery, modulus of continuity, and Gaussian white-noise style formulations.

Names worth retaining in Paper 2 notes or appendix:
- Tsybakov
- Donoho
- Donoho and Liu
- Korostelev
- Bertin
- Brown and Low
- Polyanskiy and Wu

## Paper 2 Editorial Identity

### Core message

Paper 2 should feel like the discovery that the exponent is regime-dependent because temporal roughness changes how stale evidence accumulates.

The emotional center of the paper is:
`Paper 1 found expiration. Paper 2 shows that the expiration rate depends on temporal path geometry.`

### Main conceptual sentence

`Useful memory has temporal path geometry: under smoother or rougher drift classes, staleness accumulates at different rates and induces different optimal memory scales.`

### Scope discipline

Do not let the paper drift into:
- detector engineering;
- empirical benchmark sprawl;
- weak stochastic analogies presented as theorems;
- premature sharp-constant claims.

## Proposed Manuscript Structure

Recommended structure:

1. Introduction
2. Problem Setup and Holder Path Classes
3. Upper Bound: Noise-Staleness Trade-off under Holder Drift
4. Lower Bound: Endpoint Hardness via Bump Construction and Le Cam
5. Regime Law and Special Cases
6. Constants, Tightness, and What Remains Open
7. Numerical Validation of the Scaling Law
8. Discussion and Future Directions

If a shorter structure is needed:

1. Introduction
2. Holder Horizon Law
3. Lower Bound and Tight Exponents
4. Constants and Numerics
5. Discussion

## Section-by-Section Seed

### 1. Introduction

Must establish:
- Paper 1 gave the worst-case cube-root member;
- cube-root is not universal;
- the true object is a family indexed by temporal regularity;
- this paper studies deterministic Holder-H classes;
- exponent-tight theory is proved;
- sharp constants remain open.

The introduction should not spend its first paragraph on ADWIN, UMR, or online regulation.

### 2. Problem Setup and Holder Path Classes

Need:
- formal path-class definition;
- endpoint risk definition;
- estimator class definition;
- notation for `H`, `zeta`, `sigma`, and memory length `n`;
- explicit distinction between deterministic envelope and stochastic motivation.

### 3. Upper Bound

Need:
- clean derivation of the `zeta n^H` staleness term;
- clean optimization over `n`;
- exact dependence on constants where possible;
- optional remark on EWMA or weighted windows only if it helps rather than distracts.

### 4. Lower Bound

Need:
- hard pair or bump construction;
- KL calculation;
- explicit scaling match;
- precise regime/scope statement;
- honest constant discussion.

### 5. Regime Law and Special Cases

This section should make the paper memorable.

Need:
- theorem summary box if useful;
- `H=1` as cube-root;
- `H=1/2` as square-root;
- maybe a compact phase diagram of exponent versus `H`.

### 6. Constants, Tightness, and Open Problem

Need:
- clear statement that exponents are tight;
- constants are not sharp;
- include numerical table or figure contrasting lower bound, theorem upper, exact uniform, best observed linear;
- explain why this is an open optimal-recovery-type problem rather than a defect in the main theorem.

### 7. Numerical Validation

Purpose:
- support the scaling law;
- support the constants discussion;
- illustrate special cases.

This should not turn into a large benchmark paper.

### 8. Discussion

Need:
- deterministic-vs-stochastic scope;
- relation to Paper 1;
- why sharp constants are future work rather than blocker;
- possible future work on online regime estimation and adaptive controllers.

## Figures and Tables

### Required figures

1. `Path Geometry to Horizon Law` conceptual figure.
   Content:
   - compare linear staleness (`H=1`) with rougher/smoother envelopes;
   - show how the optimal horizon changes with `H`.

2. `Upper-Lower Exponent Match` figure.
   Content:
   - log-log curves showing matched slope across `H` values.

3. `Hard Pair / Bump Witness` figure.
   Content:
   - two Holder paths with small KL and endpoint gap.

4. `Special Cases` figure.
   Content:
   - highlight `H=1` cube-root and `H=1/2` square-root inside the same family.

5. `Constant Gap` figure or table.
   Content:
   - `C_LB`, `C_UB(thm)`, `C_U(exact)`, `C_best-linear`.

### Required table

One compact table is enough if it carries the constants story.

Suggested columns:
- `H`
- `C_LB`
- `C_UB(thm)`
- `C_U(exact)`
- `C_best-linear`
- ratio `C_LB / C_best-linear`

## What Paper 2 Must Explicitly Say About Paper 1

Paper 2 should treat Paper 1 as the `H=1` worst-case member.

Recommended wording:
`The cube-root horizon law studied in the worst-case Lipschitz setting is the H=1 member of a broader Holder-parameterized family.`

That makes the papers complementary rather than revisionist.

## What Paper 2 Must Not Do to Paper 1

Do not frame Paper 1 as mistaken.

Do not say:
- cube-root was an accident;
- the original law was too narrow to matter;
- Paper 1 should have waited for Paper 2.

Correct framing:
Paper 1 established the correct worst-case foundation. Paper 2 expands the universe of path classes.

## Research Tasks Still Needed

### A. Formal theorem polishing

Tasks:
- choose exact path-class notation and metric;
- rewrite theorem statements with explicit sigma dependence;
- choose the precise lower-bound scope statement;
- remove misleading fBm phrasing;
- remove any overstrong `H -> 0` statements.

Exit criterion:
all theorem statements are manuscript-ready and venue-safe.

### B. Proof cleanup

Tasks:
- clean upper proof so the `n^H` term is transparent;
- clean lower proof so the bump width and KL scaling are transparent;
- verify constants and regime assumptions carefully;
- separate exponent-tight argument from constant-sharp discussion.

Exit criterion:
the proofs are auditable and easy to summarize.

### C. Numerical package cleanup

Relevant current scripts:
- `scripts/paper2_closed_proof.py`
- `scripts/paper2_holder_proof.py`
- `scripts/lower_bound.py`
- `scripts/numerical_validation.py`
- `scripts/tight_constant_investigation.py`

Tasks:
- verify which scripts produce theorem-supporting plots;
- standardize notation in outputs;
- separate exponent-validation outputs from constant-gap outputs;
- avoid heavy searches that do not materially change the paper.

Exit criterion:
there is a lean, credible numerical appendix and figure set.

### D. Manuscript seeding

Tasks:
- draft theorem-first introduction;
- draft notation/setup section;
- draft constants/open-problem section;
- design the main phase diagram or law summary figure.

Exit criterion:
Paper 2 can be written as a real manuscript rather than a research memo.

## Nice-to-Have but Not Blockers

- stronger numerical exploration of linear kernels;
- more refined constant tables across more `H` values;
- early attempts at online `H` estimation;
- exploratory adaptive controller discussion.

These are useful, but they should not delay the core theorem paper.

## Things That Should Be Deferred to a Later Project

- sharp minimax constants as a full closure problem;
- full stochastic-process transfer to fBm or related models;
- online regime-estimation algorithms;
- geometry-adaptive closed-loop controller;
- broad empirical systems paper showing adaptive gains across many backends.

These can become a Paper 3 or later branch.

## Suggested Writing Order

1. write the theorem statements cleanly;
2. write the lower-bound scope paragraph cleanly;
3. write the special-cases section;
4. write the constants-and-open-problem section;
5. only then write the introduction and discussion.

This is the safest order because the paper's credibility depends on theorem-level precision.

## Definition of Done

Paper 2 is ready at a high level when all of the following are true:

- the theorem is clearly deterministic Holder-H, not mislabeled as stochastic Hurst;
- upper and lower bounds match in exponent;
- `sigma` and `zeta` dependence are stated symmetrically;
- `H=1` and `H=1/2` are explicit and persuasive;
- the constants gap is discussed honestly but non-defensively;
- the paper clearly states that sharp constants remain open;
- the reader leaves convinced that useful memory has a regime-dependent geometry.

## Final Strategic Verdict

Paper 2 does not need sharp constants to be a high-level paper.

What it needs is:
- clean theorem statements;
- clean scope discipline;
- honest open-problem positioning;
- a strong conceptual message.

The right final identity is:
`a theorem paper on regime-dependent memory horizons under temporal Holder geometry, with tight exponents and an open sharp-constants frontier.`

That is already a serious paper.
