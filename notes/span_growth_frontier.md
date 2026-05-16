# Span Growth Frontier

## What the lab checked

We pushed the existing `span-growth` diagnostic in the Gaussian glue-theorem lab.

## Empirical picture

Using `n = 32, 64, 128, 256, 512` and `growth_base_span = 0.2`, the estimated
`predicted_error_exponent = 0.5 - kappa_growth_exponent` drops as the growth exponent
`beta` increases.

Observed refined sweep:

- `beta = 0.20` -> `predicted_error_exponent ~ 0.497`
- `beta = 0.30` -> `predicted_error_exponent ~ 0.490`
- `beta = 0.40` -> `predicted_error_exponent ~ 0.473`
- `beta = 0.50` -> `predicted_error_exponent ~ 0.443`

## Interpretation

The fixed-span root-`n` carrier is stable.
Span growth starts to degrade the effective exponent once the window span expands fast enough,
with the visible crossover appearing around `beta in [0.3, 0.5]` in this sweep.

This does not prove the exact threshold, but it gives a concrete boundary target:

- fixed span: theorem looks stable;
- growing span: the constant is no longer stable, and the carrier exponent begins to fall below `1/2`.

## Next frontier

The natural next test is to vary `growth_base_span` at fixed `beta` to see whether the crossover is
driven mainly by the exponent or also by the base span constant.

## Base-span sweep

Holding `beta = 0.4` fixed and varying the base span gives:

- `base_span = 0.05` -> predicted exponent `~ 0.497`
- `base_span = 0.10` -> predicted exponent `~ 0.489`
- `base_span = 0.20` -> predicted exponent `~ 0.473`
- `base_span = 0.40` -> predicted exponent `~ 0.446`
- `base_span = 0.80` -> predicted exponent `~ 0.414`

So the effective rate degradation depends on both the growth exponent and the base span.
Larger windows push the carrier further below `1/2` even at the same growth exponent.
