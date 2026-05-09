# PLAN

This is the active execution plan for the standalone tracking manuscript.
Older plans tied to the hybrid masking paper are superseded.

## Working Rule

- Keep exactly one phase in `in_progress`.
- Do not start a later phase before the current one is complete.
- Update manuscript text, figures, and claims together.
- Remove stale material instead of trying to reinterpret it if it no longer fits
  the paper.

## Active Program: Standalone Tracking Paper

### Phase 1: Scope Lock

Status: `done`

Deliverables:

- thesis sentence;
- non-goals sentence;
- narrative spine;
- minimum empirical package.

Why this phase exists:

- the paper must stop being a reduced version of the previous manuscript and
  become a standalone theorem-first tracking paper.

### Phase 2: Frontmatter Rewrite

Status: `done`

Tasks:

- rewrite `frontmatter/abstract.tex` around the tracking law;
- rewrite `frontmatter/introduction.tex` around finite-memory tracking,
  staleness, and adaptive window choice;
- remove masking-centered framing from the frontmatter.

### Phase 3: Theory Trim and Repair

Status: `in_progress`

Tasks:

- retain only the theory needed for the law and the lower bound;
- fix any sign or notation issues that affect the retained results;
- remove triadic, ISS, and masking theory from the core path of the paper.

### Phase 4: Empirical Redesign

Status: `pending`

Tasks:

- center the empirical story on the `U`-curve and finite-memory trade-off;
- add or surface ablations over drift strength and window length;
- keep EWMA and other fair finite-memory comparators;
- remove masking-centric and logged-intervention evidence from the main paper.

### Phase 5: Positioning and Close

Status: `pending`

Tasks:

- rewrite `discussion/related_work.tex` around dynamic regret,
  nonstationarity, OT, and AoI;
- rewrite `discussion/conclusion.tex` so it lands on the law and the
  freshness interpretation;
- remove residual language that points back to the cut paper.

### Phase 6: Validation

Status: `pending`

Tasks:

- rebuild the manuscript;
- rerun only the experiments still cited in the text;
- verify that captions, tables, and prose match the retained evidence.

## Current Standard

The final manuscript should make one central case:

- finite-memory tracking under drift has an unavoidable trade-off;
- this yields a cube-root optimal memory law and a finite-memory floor;
- the result can be interpreted through information freshness;
- the experiments are selected to validate that story directly.
