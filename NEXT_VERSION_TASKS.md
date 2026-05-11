# Next Version Tasks

## Priority 0: Freeze the Paper 1 Identity

1. Audit the frontmatter again.
Ensure the abstract and introduction never imply that cube-root is universal or
that the synthetic Gaussian experiments prove more than the finite-memory
trade-off and the existence of a useful-memory optimum.

2. De-center ADWIN in the empirical narrative.
Keep ADWIN as one wrapper instance, but ensure captions, tables, and summary
paragraphs present UMR as a backend-agnostic horizon cap.

3. Keep the future-work bridge narrow.
The bridge should mention narrower path classes and alternative staleness growth
only as an open problem. It should not mention Hurst, fBm, square-root, or any
new family notation.

## Priority 1: Tighten Paper 1 Technically

1. Make the $\beta$-generalization cleaner.
The current carrier-dependent remark is useful, but it should be checked for
clarity and consistency against the theorem numbering and notation.

2. Tighten the EWMA scope statement.
Keep the analogy if it helps, but make sure the paper never suggests a theorem
that it does not prove.

3. Clarify constants and calibration.
Improve the discussion of $C_K$, the role of the stationary prefix, and the
impact of the measurement layer without overselling robustness.

4. Review lower-bound notation.
Make the relation between the lower-bound constant written in terms of $\sigma$
and the upper-bound term written in terms of $C_K$ completely transparent.

## Priority 2: Clean the Empirical Story

1. Re-check all figure captions.
Any caption that sounds like "cube-root validation" should be rewritten to say
exactly what the figure establishes.

2. Re-check all summary paragraphs.
The empirical takeaway should be: finite-memory optimum, cap-only regime,
recoverability asymmetry, backend dependence.

3. Keep Bikes as appendix-only unless there is a strong reason to pull it up.
If it stays visible, add dataset provenance.

4. Decide whether to keep detector-heavy tables in their current prominence.
If they distract from the horizon-control story, push them down or rewrite the
accompanying text.

## Priority 3: Bibliography and Provenance

1. Add one direct adaptive-filtering reference if needed.
This is only necessary if the related-work paragraph on EWMA/RLS/Kalman remains
broad.

2. Add Bikes provenance if Bikes remains cited as a concrete public-stream case.

3. Decide whether to cite `MenaWeed2019` in the Sinkhorn calibration discussion.

4. Remove unused bibliography entries only after the framing is stable.

## Priority 4: UMR Implementation Hygiene

1. Keep the code comments and docstrings aligned with the paper.
UMR should read as a temporal-validity cap, not as a detector trick.

2. Check naming across experiment code.
Where possible, describe ADWIN integration as a wrapper instance rather than the
defining form of UMR.

3. Avoid introducing any regime-inference or path-geometry code into the main
branch of Paper 1 work.

## Priority 5: Validation Before Release

1. Build the manuscript.
Run `tectonic main.tex` and confirm that no new warnings or cross-reference
issues appear.

2. Run the numerical suite.
Run `uv run --with numpy python3 scripts/numerical_validation.py` and confirm
that the theorem checks still pass.

3. Do one final grep pass for overclaim language.
Examples: universal, validate cube-root scaling, law of memory, future paper,
regime-dependent, Hurst, square-root.

## Explicitly Deferred

1. Hurst-parameterized scaling family.
2. Fractional Brownian / path-geometry theorem statements.
3. Lower bounds for $H < 1$.
4. Online Hurst estimation.
5. Native UMR tracker with regime inference and resolvability gates.

These are part of the broader research program and should not be folded into the
current paper version.
