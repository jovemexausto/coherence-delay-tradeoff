# PAPER 2

## Identity

Working title:
`Useful Memory Has Temporal Path Geometry`

Alternative titles:
- `A Holder-Parameterized Horizon Law for Finite-Memory Tracking`
- `Temporal Path Geometry of Useful Memory Under Drift`
- `Regime-Dependent Horizon Laws for Finite-Memory Tracking`

One-sentence thesis:
The exponent governing the useful-memory horizon is not universal; it depends on temporal path geometry, and deterministic Holder-type path classes induce regime-dependent optimal memory and minimax risk laws.

One-sentence reviewer summary:
Paper 1 gives the worst-case Lipschitz member of the story. Paper 2 shows that this is one member of a larger family: if staleness accumulates like `n^H`, then the optimal horizon and minimax risk scale with `H`, and the upper and lower exponents match.

Core technical message:

```tex
\mathcal E(n)\le C_K n^{-1/2}+c_H\zeta n^H
```

so that

```tex
n^*(H,\zeta)\asymp (C_K/\zeta)^{2/(1+2H)},
\qquad
R^*(H,\zeta)\asymp \sigma^{2H/(1+2H)}\zeta^{1/(1+2H)}.
```

Core conceptual message:
`Temporal roughness determines how staleness accumulates, and therefore determines the horizon of useful memory.`

## Non-Negotiable Identity

Paper 2 is about:
- deterministic temporal-regularity classes;
- regime-dependent useful-memory horizons;
- the family of exponents indexed by temporal roughness;
- the unification of Paper 1's cube-root case with rougher and smoother classes;
- the idea that staleness accumulation depends on path geometry.

Paper 2 is not about:
- ADWIN;
- detector wrapping;
- a vague story about many regimes without theorem-level closure;
- fractional Brownian motion as the literal theorem object;
- sharp constants as a prerequisite for publication.

## Central Reframe

The wrong version of Paper 2 is:
`Paper 1, but generalized a bit`

The right version of Paper 2 is:
`the rate at which memory expires depends on temporal path geometry`

Paper 2 exists to change the ontological status of the horizon.

After Paper 1, the horizon looks finite.
After Paper 2, the horizon looks geometric.

## Reader Effect

Target reader reaction:
`Paper 1 gave the worst-case envelope. Paper 2 shows that this envelope is one member of a theorem-level family indexed by temporal roughness.`

Target closing intuition:
`Useful memory does not decay at one universal rate. Different path classes make staleness accumulate differently.`

If the reader instead concludes:
- `this is really about fBm folklore`;
- `this is a loose regime narrative without closure`;
- `this is just Paper 1 with a parameter inserted`;

then Paper 2 has failed.

## Official Language

Use consistently:
- temporal path geometry
- deterministic Holder class
- temporal roughness
- regime-dependent horizon law
- regime-dependent minimax risk
- staleness accumulation
- special-case recovery of cube-root

Use carefully:
- Hurst-type motivation
- stochastic motivation

Avoid in theorems and core claims:
- direct claims about fractional Brownian motion unless actually proved;
- language that blurs deterministic envelopes with stochastic-process laws;
- vague `many regimes` rhetoric without a precise class.

## Relationship to Paper 1

Paper 1 says:
`under the worst-case Lipschitz envelope, memory has a finite horizon and can become stale before detectable change`

Paper 2 says:
`the Lipschitz case is only one member of a larger family, because the rate of staleness accumulation depends on temporal path geometry`

Paper 1 is about expiration under the worst-case envelope.
Paper 2 is about the geometry of expiration across path classes.

Paper 2 must not weaken Paper 1 by making it look provisional or naive.
It must instead make Paper 1 look like the `H=1` cornerstone.

## The Main Mathematical Object

Paper 2 should be stated for deterministic Holder-type classes of the form

```tex
\mathcal H(H,\zeta)
=
\left\{(P_t^*)_t : d(P_t^*,P_s^*)\le \zeta |t-s|^H \text{ for relevant } s,t\right\}.
```

This class should be interpreted as an envelope model for temporal roughness.

That means:
- the theorem is deterministic;
- the object is a path class, not a stochastic-process construction;
- Hurst language is motivational at most;
- the theorem should stand without fBm.

## Theorem Package

Paper 2 should revolve around a clean package:

1. Upper bound for finite-memory tracking under Holder-type drift.
2. Optimized horizon and minimized risk.
3. Lower bound with matching exponent.
4. Explicit recovery of special cases, especially `H=1` and `H=1/2`.

The editorial payoff is that cube-root and square-root stop looking unrelated.
They become members of the same geometric family.

## Closed vs Open

Closed enough for the paper's main claim:
- the deterministic Holder-class framing;
- an upper law with staleness term `n^H`;
- a lower bound with matching exponent;
- the scaling exponents `2/(1+2H)` and `1/(1+2H)`;
- the special cases that recover known-looking laws.

Still open and should be stated honestly:
- sharp constants;
- exact extremal path structure;
- exact extremal estimator characterization;
- any stronger stochastic-process interpretation.

The open problem is not the exponent family.
The open problem is the sharp minimax geometry inside that family.

## Narrative Arc

### Act 1: Beyond the Worst Case

Paper 1 identifies the worst-case Lipschitz envelope.
Paper 2 begins by saying that this cannot be the end of the story, because worst-case linear staleness is only one way that age can accumulate.

### Act 2: Roughness Determines Expiration Rate

If temporal misalignment grows like `n^H`, then the variance-staleness trade-off changes accordingly.

This is the central move.

The paper should make the reader feel that the horizon exponent is not mysterious; it is the consequence of how temporal roughness accumulates with lag.

### Act 3: The Family of Horizon Laws

The reader should then see:
- one family of upper laws;
- one family of optimized horizons;
- one family of minimax rates.

This is where Paper 2 earns the phrase `temporal path geometry`.

### Act 4: Special Cases as Editorial Payoff

`H=1` should recover Paper 1.
`H=1/2` should recover the square-root case.

These are not afterthoughts.
They are the reader's proof that the family is real.

### Act 5: Honest Boundary

The ending should make clear:
- the exponent family is solved at theorem level;
- the sharp constants are not;
- stochastic interpretations remain motivational unless proved.

## Conceptual Structure

If born correctly, Paper 2 would read conceptually as:

1. Introduction
2. Deterministic Holder Classes as Temporal-Roughness Envelopes
3. Upper Horizon Law
4. Lower Bound and Matched Exponents
5. Special Cases and Geometric Interpretation
6. Limits and Open Problems
7. Conclusion

This is a statement of intellectual structure, not of implementation.

## Opening Standard

The opening must do exactly this:
- state that the horizon exponent is not universal;
- state the deterministic Holder-class object;
- state the regime-dependent upper and lower laws;
- explain that Paper 1 is recovered as `H=1`;
- state honestly that sharp constants remain open.

It must not:
- oversell fBm;
- sound like a speculative geometry manifesto;
- rely on future work to justify the present theorem;
- bury the theorem behind broad motivation.

## Theoretical Standard

The theory must protect:
- a clean deterministic class definition;
- a clean upper theorem;
- a clean lower theorem with matching exponent;
- symmetric scaling statements;
- explicit scope around what is and is not sharp.

The theory must avoid:
- imprecise stochastic claims;
- fake generality;
- rhetorical overreach about universality.

## Figure Standard

Paper 2 should be figure-light but conceptually sharp.

The figures that matter most are:
1. a regime-family conceptual figure showing how staleness growth changes with `H`;
2. a scaling figure for horizon and risk exponents across `H`;
3. a lower-bound witness figure adapted to the Holder setting.

Any figure should answer a conceptual question, not just decorate algebra.

## Reviewer Attack Surface

### Attack
`This is really about fractional Brownian motion, but you did not prove that.`

Response:
No. The theorem is about deterministic Holder envelopes. Stochastic roughness families are only motivation unless explicitly proved.

### Attack
`The constants are not sharp.`

Response:
Correct. The exponent family is the closed theorem-level result; the sharp constants are the next problem.

### Attack
`Is this just Paper 1 with a parameter H inserted?`

Response:
No. The conceptual claim changes: the horizon is no longer a single worst-case law but a family indexed by temporal roughness.

### Attack
`Why should anyone care about this family?`

Response:
Because it explains when cube-root is the right law and when it is not. It upgrades a single worst-case statement into a geometry of horizon laws.

## Definition of Done

Paper 2 is ready only when all of the following are true:
- the horizon exponent is clearly presented as non-universal;
- the deterministic Holder-class object is clean and defensible;
- the upper and lower exponents match;
- Paper 1 is visibly recovered as `H=1`;
- the paper is honest about constants and stochastic interpretation;
- a reviewer can summarize the work as `useful memory has temporal path geometry`.

## Final Positioning

Paper 2 is:
`the theorem-level geometry of useful-memory horizons across temporal roughness classes.`

It should feel like a genuine expansion of the object, not a parameterized sequel.
