# Short-Paper Contract

## Paper contract

This project is a short scientific paper about the temporal-validity horizon for
finite-memory distribution tracking under drift and the useful-memory region
induced by that horizon.

The main text must do only five things:

- define the object
- state the general horizon law
- close one tractable proof model
- show that the horizon is structural
- delimit benchmark, conjectural extension, and empirical signatures

The main text is successful when a reader can identify, without ambiguity:

- the central statistical object
- the main theorem line
- which results are closed
- which results are conjectural
- what the empirical section is evidence for

The main text is not a repository inventory, a project history, or a provenance
dump.

## Claim ledger

### Theorem

- Abstract upper law
- Temporal-validity horizon
- Useful-memory region induced by the horizon law
- Uniform-window staleness bound with exact finite-`n` constant `C_{H,n}`
- Tractable 1D bounded-support fixed-span triangular-array proof model with root-`n` finite-sample rate
- Structural Gaussian lower bound at the exponent level for the root-`n` regime

### Benchmark result

- Gaussian location minimax benchmark on deterministic Holder paths
- Compact Gaussian lower-bound constants proposition

### Conjecture

- Fixed-`epsilon` Sinkhorn horizon inheritance on the embedded fixed-span model
- Regular-family horizon inheritance

### Open problem

- Useful-memory design beyond uniform weights
- Online useful-memory adaptation
- Validity-detection theory

## Paper / appendix / repo split

### Main text

- object, law, useful-memory region, proof model, structural lower bound, Gaussian benchmark
- explicit conjectures with clear status labels
- empirical signatures directly tied to the object or to a named conjecture
- limitations and open problems

### Appendix

- only technical support needed for readability of the paper
- short repository guide mapping claims to code, artifacts, and notes
- no large calibration tables unless indispensable for interpreting a claim in the main text

### Repository

- provenance and regeneration map
- calibration tables and sweeps
- extended diagnostics
- constant refinements and exploratory notes
- artifact generation scripts and tests
- claim-to-artifact traceability

## Banned from main text

- project history
- framing evolution
- justification for why sections exist
- repository/provenance detail beyond a minimal pointer
- extended calibration grids
- proof-gap side results
- side corollaries not needed for the theorem line
- rhetoric built from `canonical/noncanonical`
- terms such as `carrier`, `minimum kernel`, and similar internal scaffolding
- any claim whose status is not explicit

## Editing rule

When deciding whether to keep a sentence in the main text, ask:

1. does it strengthen the object?
2. does it strengthen the theorem line?
3. does it strengthen claim-status clarity?

If the answer to all three is no, it does not belong in the main text.
