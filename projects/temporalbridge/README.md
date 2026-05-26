# TemporalBridge

This workspace isolates the spin-off surface for the temporal-validity horizon
controller before consolidation into the root manuscript.

## Layout

- `code/temporalbridge/`: dedicated Python package for the spin-off
- `tests/`: dedicated spin-off tests
- `notes/`: local planning notes for the controller and API surface

## Current scope

The initial package layer is intentionally thin. It reuses the validated lag-
geometry and bootstrap machinery from `projects/scale-consistency/code` and
exposes a cleaner scientific core:

- `fit_horizon`
- `bootstrap_horizon`
- `calibrate_alarms`
- `detect_alarms`
- `validity_controller`

Adapters and convenience wrappers live behind that core.

## Local test command

Run from this directory with both code roots on `PYTHONPATH`:

`PYTHONPATH=code:../scale-consistency/code python -m unittest discover -s tests`
