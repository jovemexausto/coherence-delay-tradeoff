# Roughness + Span-Growth Frontier

## What was tested

We combined the cusp-kernel roughness lab with span growth:

`span_n = base_span * n^beta`

and measured Bahadur remainder rates for `alpha \in {1.0, 0.5, 0.25}`.

## Key sweep

For `beta = 0.0, 0.25, 0.5, 0.75` the remainder still stayed above the root-`n` threshold,
but it degraded sharply by `beta = 0.75`.

For `beta = 1.0` the regime broke clearly:

- `alpha = 1.0` -> residual rate `~ 0.257`
- `alpha = 0.5` -> residual rate `~ 0.248`
- `alpha = 0.25` -> residual rate `~ 0.132`

## Interpretation

This is the first frontier where the proof story genuinely fails:

- roughness alone does not kill the root-`n` remainder in the bounded-support interior-band lab;
- span growth alone already degrades the exponent;
- together, sufficiently fast span growth (`beta = 1.0`) pushes the remainder well below the `1/2` barrier.

## Takeaway

The minimum theorem is safe in the fixed-span regime, but the combined roughness + span-growth frontier is the real boundary where the Bahadur remainder proof stops being stable.
