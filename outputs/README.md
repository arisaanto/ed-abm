# Outputs

`outputs/latest/` holds the current imported reference result for local inspection. `outputs/batch/` is for local smoke/batch runs and can be cleared when a run is obsolete. `outputs/findings/` is reserved for durable presentation-ready findings only.

## Current Latest Outputs
- `latest/ABM_results/part1_validation_n100/`: canonical frozen Part 1 n100 reference import.

## Durable Findings
Write to `outputs/findings/` only after a result is good enough to show. Stale Part 1/Part 2 findings from older framing should be deleted or replaced when the baseline is reworked.

## Normal Commands
- `python3 main.py --verify`
- `python3 main.py --mechanism-audit --duration 14400 --n-runs 1`
- `python3 main.py --validation-candidate --validation-target care_area --duration 43200 --n-runs 3 --warmup-seconds 7200`
- `python3 main.py --condition-smoke --duration 600`
- `python3 main.py --part2-scenarios --duration 43200 --warmup-seconds 7200 --n-runs 3 --conditions baseline,cockpit_only,nursta_only,both`

The normal workflow uses one rule-based perception baseline. SeniorDoctor and Part 3 generative cognition are disabled unless explicitly requested for sensitivity or future extension work.
