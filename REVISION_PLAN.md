# Revision Plan

Historical note:

- This document records the revision path that produced the current standalone drift paper.
- The manuscript identity is now locked as `Useful Memory Has a Horizon: A Cube-Root Law for Tracking Under Drift`.

## Objective

Turn the current manuscript into a standalone paper on useful memory under drift and the coherence-lag trade-off that governs it.

## Final Paper Stance

- agents with finite memory must track a drifting target distribution;
- larger windows reduce statistical noise but increase informational staleness;
- this induces a cube-root optimal memory law;
- a minimax lower bound shows an unavoidable finite-memory floor;
- temporal coherence is the main operational object rather than a generic freshness story.

## What Stays

- finite-memory tracking under Wasserstein drift;
- the cube-root window law;
- the finite-memory floor and lower bound;
- the operational horizon-regulation consequence of the law;
- Gaussian and related tracking experiments that directly support the law;
- EWMA or similar finite-memory comparators;
- a narrow related-work bridge to dynamic regret and nonstationary estimation;
- UMR as the first operational consequence of the law.

## What Goes

- coercive masking as a central theme;
- `CI^E` as a flagship result;
- RLHF / Goodhart as manuscript framing;
- intervention-aware diagnosis as a primary contribution;
- KuaiRand and particle masking as central evidence;
- any claim that passive-detector comparisons establish universal superiority;
- any framing that makes drift detection the central object instead of memory validity.

## Narrative Spine

1. Pose the tracking problem under nonstationary drift.
2. Explain the finite-memory trade-off between estimation error and staleness.
3. State the cube-root law and the resulting error floor.
4. Show the lower bound for the restricted estimator class.
5. Interpret the result as a temporal-coherence law.
6. Validate the law with experiments and ablations designed for this story.
7. Show that the same law can be turned into a usable horizon regulator.

## Bottom Line

The job is to turn the surviving tracking contribution into a focused standalone paper on useful memory under drift, with its own narrative and evidence.
