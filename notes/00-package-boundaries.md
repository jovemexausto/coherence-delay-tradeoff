# 00. Package Boundaries
Status: active
Category: meta
Prev: -
Next: 01. Main Theorem Package

This note fixes the project boundary between:

- the closed paper core;
- supporting benchmark material;
- explicit conjectures;
- repository-only provenance and diagnostics.

## Closed paper core

- abstract upper law;
- optimal temporal-validity horizon;
- useful-memory region induced by the horizon law;
- exact finite-`n` uniform-window staleness constant;
- tractable 1D bounded-support fixed-span proof model with root-`n` finite-sample rate;
- structural Gaussian lower bound at the exponent level;
- Gaussian location minimax benchmark.

These results are enough to support the central scientific claim.

## Supporting benchmark material

- compact Gaussian lower-bound constants proposition;
- empirical signatures: U-curve, roughness scaling, useful-memory region, validity-detection lag;
- appendix-level finite-sample diagnostics.

These strengthen interpretation or sharpness without changing the main theorem line.

## Explicit conjectures

- fixed-`epsilon` Sinkhorn horizon inheritance on the embedded fixed-span model;
- regular-family horizon inheritance.

These may appear in the paper only with explicit conjecture status.

## Repository-only provenance

- large calibration sweeps;
- extended diagnostics;
- exploratory constant audits;
- implementation-level provenance beyond what is needed for reading the paper.

## Main-text rule

Something belongs in the main text only if it strengthens at least one of:

- the central object;
- the theorem line;
- claim-status clarity.

Otherwise it belongs in the appendix or repository provenance layer.
