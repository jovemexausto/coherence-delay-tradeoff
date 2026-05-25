# Sequential Validity versus Detection Delay

Short abstract: temporal invalidity can arise before standard changepoint alarms, so the relevant operational gap is the delay between horizon loss and detector response.

## Seed

This project starts from the validity-before-detection signature already observed in the main paper. The focus here is the sequential-theory layer: quantify the gap, identify when it persists across detectors, and isolate which assumptions are needed for a real delay law.

## Scope

- validity horizon versus detection delay
- observation-based versus residual-based detector inputs
- detector-family robustness and tuning sensitivity
- lag geometry as a sequential quantity

## Initial questions

1. When does the sign of the validity-detection gap persist across detectors?
2. What assumptions are needed for a time-uniform delay statement?
3. Which detector families can be compared on the same drift geometry?
4. What is the simplest theory/empirics split that keeps the paper tight?

## Suggested structure

- `main.tex`
- `frontmatter/abstract.tex`
- `frontmatter/introduction.tex`
- `theory/main_law.tex`
- `discussion/limitations.tex`
- `discussion/conclusion.tex`
- `notes/working-plan.md`
