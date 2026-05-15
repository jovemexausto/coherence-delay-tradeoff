# PAPER 1

## Identity

Title:
`Useful Memory Has a Horizon`

Subtitle:
`Structural Worst-Case Limits for Finite-Memory Tracking under Drift`

One-sentence thesis:
Under worst-case drift, memory is both a statistical resource and a temporal liability, so finite-memory systems can become stale before change is detectable.

One-sentence reviewer summary:
This paper shows that under a worst-case Lipschitz drift envelope, finite memory has a finite useful-memory horizon, and that temporal validity can fail before a changepoint detector has enough evidence to alarm.

Core technical claim:

```tex
\mathcal E(n)\le C_K n^{-1/2}+\frac12\zeta n
```

and therefore

```tex
n^*\asymp (C_K/\zeta)^{2/3},
\qquad
\mathcal E_{\min}\asymp C_K^{2/3}\zeta^{1/3}.
```

Core conceptual claim:
`Temporal validity is not changepoint evidence.`

Core empirical claim:
`Finite memory can become stale before change becomes detectable.`

## Non-Negotiable Identity

Paper 1 is about:
- finite-memory tracking under drift;
- temporal validity of retained evidence;
- a worst-case useful-memory horizon;
- a subclass-based finite-memory floor;
- detector-silent staleness as the main practical failure mode.

Paper 1 is not about:
- UMR;
- ADWIN improvement;
- a detector wrapper;
- a regulator as a contribution;
- a benchmark suite;
- a universal law of memory;
- a disguised transition paper toward Paper 2.

If some part of the paper matters only because a named adaptive mechanism exists, that part does not belong to Paper 1 in its present form.

## Central Reframe

The wrong version of Paper 1 is:
`we have a mechanism that improves adaptation under drift`

The right version of Paper 1 is:
`finite memory becomes stale before change becomes detectable`

Everything in the paper must serve that sentence.

The theory explains why the phenomenon exists.
The empirical section makes the phenomenon legible.
No method should be carrying the paper.

## What Must Be Purged

The following are contamination, not assets:
- the name `UMR`;
- any contribution bullet whose subject is a mechanism;
- any suggestion that the paper proposes a regulator;
- `ADWIN + something` as a protagonist;
- any benchmark rhetoric that turns the paper into a comparison suite;
- any systems flavor that makes the reader think the contribution is implementation.

This is not a matter of de-emphasis.
It is a matter of identity.

## Reader Effect

Target reader reaction:
`This paper isolates a structural failure mode of finite memory under drift: stale evidence can persist before there is enough evidence for a detector to fire.`

Target closing intuition:
`More memory is not always more information. Under drift, remembered evidence has an expiration date.`

If the reader instead concludes:
- `this is a detector paper`;
- `this is a wrapper paper`;
- `this is a benchmark paper with theory on top`;

then the paper has failed.

## Official Language

Use consistently:
- useful-memory horizon
- temporal validity
- staleness
- finite-memory floor
- worst-case Lipschitz envelope
- detector-silent staleness
- temporal misalignment
- horizon misalignment

Avoid:
- UMR
- regulator
- wrapper
- booster
- controller
- backend-agnostic
- cap-only regime in final prose if a better thesis-level phrase exists
- universal law
- path-geometry language that belongs to Paper 2

## Narrative Arc

### Act 1: The Paradox of Memory

Opening logic:
- in stationary settings, more memory helps;
- under drift, more memory also carries obsolete evidence;
- memory is therefore statistically helpful and temporally dangerous;
- useful memory must have a finite horizon.

Sentence to preserve in spirit:
`Under drift, the question is not only how much evidence we have, but how old that evidence can be before it stops describing the present.`

### Act 2: The Worst-Case Law

The theory must do four things:
- derive the variance-staleness decomposition;
- show that the U-curve is structural;
- identify a finite optimum;
- prove that the finite-memory floor is structural rather than estimator-specific.

The theory must not:
- imply universality outside the stated class;
- quietly generalize to every adaptive memory scheme;
- preview Paper 2's broader regime family.

### Act 3: The Conflict of Clocks

Detector question:
`Is there enough statistical evidence of change?`

Temporal-validity question:
`Is the retained evidence still valid for the present?`

Main sentence:
`A detector can remain statistically silent while retained memory is already stale.`

Detectors enter only as contrast instruments.
They are not protagonists and not targets for improvement.

### Act 4: Empirical Legibility

The empirical section exists only to make the structural phenomenon visible in finite samples.

It does not need:
- a hero mechanism;
- a named adaptive policy;
- a leaderboard;
- a system contribution;
- a method-comparison identity.

If an experiment matters only because some adaptive intervention is doing all the conceptual labor, that experiment belongs to a different paper.

### Act 5: Disciplined Ending

Allowed bridge:
`The cube-root exponent is tied to the worst-case linear-staleness envelope studied here. Narrower path classes may induce different effective staleness growth, which we leave to future work.`

That is enough.

## Conceptual Structure

If born correctly, the paper would read conceptually as:

1. Introduction
2. Worst-Case Useful-Memory Law
3. Structural Finite-Memory Floor
4. Temporal Validity and Detector-Silent Staleness
5. Empirical Evidence
6. Related Work
7. Limitations and Scope
8. Conclusion

This is a statement of vision, not a statement about implementation.

## Opening Standard

The abstract and introduction must do exactly this:
- state that memory helps and ages;
- state the worst-case horizon problem;
- state the upper law and finite optimum;
- state the lower bound and structural floor;
- state the separation between temporal validity and changepoint evidence;
- close on stale memory before detectable change.

They must not:
- introduce a named method;
- sound like a detector paper;
- sound like a systems paper;
- suggest the paper is selling a cap policy.

## Theory Standard

The theory should protect four things:
- the decomposition;
- the finite optimum;
- the lower bound witness;
- the interpretation of temporal validity.

The operative line throughout is:
`cube-root is the worst-case linear-staleness-envelope result`

Nothing in the theory should read as if the paper is trying to prove a general theory of all memory laws.

## Empirical Standard

The empirical section needs only enough evidence to make the main phenomenon legible.

It should show:
1. a structural U-curve;
2. detector-silent staleness;
3. misalignment cost;
4. optionally, restrained external confirmation.

It should not need:
- a named adaptive mechanism;
- a wrapper story;
- superiority rhetoric;
- a multi-method arena as identity;
- calibration drama as a source of narrative importance.

Hard rule:
if a figure needs `UMR` in order for the reader to care, that figure is misframed for Paper 1.

## Contributions

The clean contribution list is:

- We formalize useful memory under drift as a variance-staleness trade-off in Wasserstein geometry.
- We prove a worst-case finite useful-memory horizon under local `W_2`-Lipschitz drift.
- We show a Gaussian-location lower bound establishing a structural finite-memory floor in the critical-window regime.
- We distinguish temporal validity from changepoint evidence and identify detector-silent staleness as an operational regime.
- We show empirically that stale memory can persist before detectable change, with measurable cost once horizon misalignment grows.

No contribution bullet should have a mechanism as its grammatical subject.

## Figures

The paper should be readable from figures alone.

The indispensable figures are:
1. `Two Clocks of Drift`
2. `Structural U-Curve`
3. `Lower-Bound Witness`
4. `Detector-Silent Staleness`
5. `Horizon Misalignment Cost`

Recovery asymmetry is optional.
Real-stream evidence is optional.

Figures should be cut if they:
- center a method in the legend or caption;
- read like a benchmark comparison;
- require a named policy to explain their importance.

## Secondary Material Principle

Secondary material should support scope and reproducibility.
It should not smuggle a shadow systems paper back into the manuscript.

If supporting material reads like protocol, wrapper mechanics, or deployment guidance, it is probably serving the wrong identity.

## Reviewer Attack Surface

### Attack
`Cube-root already appears elsewhere.`

Response:
Yes, but here the object is finite-memory temporal validity under distribution drift, together with a structural lower bound and a separation from detectability.

### Attack
`The lower bound is subclass-based.`

Response:
Correct, and that is sufficient because the Gaussian location subclass sits inside the ambient drift class.

### Attack
`Other path classes may give other exponents.`

Response:
Correct. This paper is explicitly the worst-case Lipschitz foundation.

### Attack
`Where is the adaptive method?`

Response:
There is no central adaptive method claim. The point of the paper is a structural failure mode of finite memory, not a new control policy.

That response should be embraced, not apologized for.

## Definition of Done

Paper 1 is ready only when all of the following are true:
- the opening states the paper's message cleanly and narrowly;
- cube-root appears only as the worst-case Lipschitz-envelope result;
- the lower bound is readable and visibly structural;
- the empirical storyline is `U-curve -> detector-silent staleness -> misalignment cost`;
- no named mechanism is needed to explain why the paper matters;
- a reviewer can summarize the paper as `finite memory becomes stale before change becomes detectable`.

## Final Positioning

Paper 1 is:
`a worst-case theory of temporal validity for finite memory under drift.`

Anything that cannot survive under that identity should be discarded without sentiment.
