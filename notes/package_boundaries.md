# Package Boundaries

This note separates the current project into proved or theorem-ready components, theorem targets, conjectures, and future work.
Its role is to stop scope drift and keep the manuscript honest.

## Proved or theorem-ready

- `Theorem 1`: the abstract upper law in `notes/abstract_upper_law.md`.
- `Corollary 1`: the optimized useful-memory scale in `notes/abstract_upper_law.md`.
- `Proposition 3`: the minimum-kernel carrier in theorem-ready form in `notes/minimum_kernel_proposition.md`.
- the consolidated minimum-kernel proof narrative in `notes/minimum_kernel_proof.md`.
- `Theorem 2` at the structural level for the main `a = 1/2` slice in `notes/structural_lower_bound.md`.

## Theorem targets for the main paper package

- `Proposition 4`: useful-layer inheritance under bounded support, fixed span, and low intrinsic dimension in `notes/useful_layer_bridge.md`.
- `Proposition 5`: practical measurement-layer inheritance for fixed-`epsilon` Sinkhorn in `notes/practical_layer_measurement.md`.

These targets can appear in the paper as explicit theorem goals or theorem-shaped statements with supporting evidence, but they should not be described as closed proofs unless they are actually written and checked at theorem level.

## Conjectures

- raw `W_2` triangular-array carrier inheritance in broader high-dimensional settings;
- fixed-`epsilon` Sinkhorn preserving a stable carrier beyond the current low-intrinsic geometry;
- roughness-matched lower-bound extensions that are class-tight for general deterministic Hölder path classes;
- universal improvements from non-uniform weighting beyond constant-level tuning.

## What belongs in the principal manuscript

- the abstract law and its optimized horizon;
- the minimum kernel as the first rigorous carrier instantiation;
- the structural lower bound for the main slice;
- the useful-layer bridge as the first extension beyond the proof kernel, if kept clearly marked by status;
- the practical-layer evidence as a measurement-layer target, if kept clearly marked by status.

## What belongs in appendix or notes

- Bahadur and Kiefer remainder details;
- design-effect calculations;
- frontier maps for span growth and roughness;
- numerical diagnostics supporting theorem ingredients;
- measurement-layer sweeps and robustness tables.

## Future work and open problems

- close the useful-layer inheritance theorem beyond the current experimental bridge;
- close the practical measurement-layer theorem;
- extend the lower bound beyond the main `a = 1/2` slice and obtain sharper constants;
- investigate non-uniform windows and whether they change constants or exponents;
- test the theory on real data and against adaptive horizon-selection baselines.

## Rule for scope decisions

If a statement is needed to justify the paper's central scientific claim, it belongs in the theorem package.
If it only strengthens the package, sharpens constants, or broadens coverage beyond the current closed slice, it belongs in conjectures, appendix, or future work.
