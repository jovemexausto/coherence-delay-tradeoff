# Rough Kernel Frontier

## What was tested

We replaced the uniform kernel in the Bahadur lab with a symmetric cusp kernel on `[-1,1]`:

`g_\alpha(z) \propto 1 + |z|^\alpha`

with `\alpha \in (0,1]`.

## Lab outcome

Across `\alpha = 1.0, 0.75, 0.5, 0.25, 0.1`:

- the full residual rate stayed above `1/2`;
- the empirical increment remained the dominant term;
- the Taylor term varied with roughness but stayed smaller than the empirical term;
- the reconstruction error stayed tiny.

## Interpretation

This is consistent with the proof story:

- rougher densities affect the Taylor piece;
- the empirical increment piece is still the main bottleneck;
- the full Bahadur remainder remains safely above the root-`n` threshold in this bounded-support interior-band regime.

## Next frontier

The remaining real stress test is to combine roughness with span growth, since that couples the two mechanisms that already look like the real boundary drivers.
