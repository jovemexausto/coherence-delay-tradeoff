# 00. Package Boundaries
Status: active
Category: meta
Prev: -
Next: 01. Main Theorem Package

This note separates the project into:

- closed paper-level results;
- sharp refinements that strengthen the canonical `a = 1/2` regime;
- theorem targets beyond the current core;
- open frontier questions.

Its role is to keep the manuscript honest while still letting the theory grow.

## Closed paper-level results

- `Theorem 1`: the abstract upper law in `notes/02-general-law.md`.
- `Corollary 1`: the optimized useful-memory horizon law in `notes/02-general-law.md`.
- the exact finite-`n` uniform-window staleness constant recorded in `notes/01-main-theorem-package.md`.
- `Proposition 3`: the minimum-kernel carrier in `notes/03-minimum-kernel-carrier.md`.
- the consolidated minimum-kernel proof narrative in `notes/04-minimum-kernel-proof.md`.
- `Theorem 2` at the structural level for the canonical `a = 1/2` regime in `notes/05-structural-lower-theory.md`.

These are enough to support the paper's central scientific claim.

## Sharp refinements of the canonical `a = 1/2` regime

- exact Gaussian ramp-witness frontier in `notes/06-exact-gaussian-witness-frontier.md`;
- witness-shape extremality in `notes/07-witness-shape-extremality.md`;

These strengthen the lower and staleness stories without changing the identity of
the paper. They are good candidates for remarks, appendix material, or a short
refinement subsection.

## Theorem forms beyond the current core

- extended-regime carrier inheritance under bounded support, fixed span, and low intrinsic dimension in `notes/08-extended-regime.md`;
- operational-regime carrier inheritance for fixed-`epsilon` Sinkhorn in `notes/09-operational-regime.md`.

These can appear in the paper only if they are kept clearly marked by status.
They are not secondary in scientific role. They are broader theorem forms of the
same carrier theory.

## Open frontier questions

- class-tight lower bounds for general deterministic Hölder path classes;
- lower bounds beyond the canonical `a = 1/2` regime;
- whether the endpoint-minimal witness is already the optimal subclass shape in a broader admissible family;
- raw `W_2` carrier inheritance beyond the current low-dimensional safe zone;
- fixed-`epsilon` Sinkhorn theorem-level carrier guarantees;
- non-uniform memory shapes beyond constant-level tuning.

## What belongs in the principal manuscript

- abstract law and optimized horizon;
- exact finite-`n` uniform-window staleness bound;
- minimum kernel;
- structural lower bound for the canonical `a = 1/2` regime;
- extended and operational regimes, with careful status labeling.

Optional but now justified if space permits:

- a remark on the exact Gaussian ramp frontier;
- a remark that the ramp witness is not shape-optimal for `H < 1`.

## What belongs in notes or appendix

- Bahadur and Kiefer remainder details;
- design-effect calculations;
- span-growth and roughness frontier diagnostics;
- numerical tables supporting theorem ingredients;
- exact Gaussian witness frontier calculations;
- witness-shape extremality calculations;
- broader sharp-constant audits for the `(a,H)` family when needed.

## Rule for scope decisions

If a statement is needed to justify the paper's central claim that useful memory
has a roughness-dependent structural horizon, it belongs in the theorem package.

If a statement only sharpens constants, improves a witness on the already-closed
canonical `a = 1/2` regime, or broadens coverage beyond the current proof kernel, it belongs in a
refinement note, appendix, or future work unless there is a strong presentational
reason to elevate it.
