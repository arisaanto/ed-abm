# Architecture

## Implemented Core

The active Part 1/2 model is non-LLM and rule-based. Staff execute explicit clinical workflow through wall-valid movement and state transitions. Opportunistic communication follows this bounded pipeline:

1. compute visible staff through condition-specific line of sight, field of view, and isovist geometry;
2. identify reason-bearing, workflow-compatible candidates;
3. apply salience, attention, pressure, interruption, and repeat-contact constraints;
4. approach only through valid movement and log an interaction only at embodied proximity;
5. keep workflow events separate from validation-counted communication.

Physical walls control movement and collision. Visibility walls control perception. They are deliberately separate so transparent partitions can alter intervisibility without deleting physical barriers.

## Part 2 Conditions

- `baseline`: the frozen care-area condition used for the development-informed
  comparison with shadowing observations. That comparison is not independent
  validation.
- `cockpit_only`: changes COCPIT transparency and visibility occlusion; physical routing geometry remains baseline.
- `nursta_only`: relocates NURSTA physical/station semantics and associated station-return and route exposure.
- `both`: combines COCPIT transparency and NURSTA relocation.

Condition invariants are implemented in `src/conditions.py`: baseline and `cockpit_only` share physical NURSTA geometry; `nursta_only` and `both` share moved NURSTA geometry; COCPIT transparency changes `visibility_walls`, not `effective_walls`.

## Sensitivity Overrides

`scripts/run/run_single.py` allowlists active behavioral overrides and applies them before `Simulation` construction within an independent one-run process. Scenario overrides therefore enter the copied per-run scenario definition without changing no-override defaults. Intended/effective values and pressure-gate counters are written to each summary. Incomplete or out-of-range scenario sensitivity arguments fail before simulation construction.

`SCENARIO_PRESSURE_ACTION_THRESHOLD` targets the existing active `pressure_action_threshold` gate. The retired `SCENARIO_PRESSURE_STATION_SOCIAL_SUPPRESSION` factor is not allowlisted: its value sat behind a deterministic earlier gate and produced identical low/high outputs.

## Part 3 Boundary

The optional Part 3 scaffold is downstream and opt-in:

- `src/personas.py`: staff role definitions, role-independent Part 3 cognitive orientations, and balanced assignment logic;
- `src/interviews.py`: explicitly synthetic survey/interview schemas;
- `src/interaction.py`: bounded rule-based interaction engine plus the bridge
  to the opt-in Part 3 controller; it does not load vLLM;
- `src/cognitive_policy.py`: shared conservative context-token estimator for Part 3 packet builders;
- `src/part3_closed_loop.py`: opt-in bounded causal controller and grounded memory policy;
- `src/vllm_backend.py`: structured offline and brokered closed-loop inference backend, imported only by explicit Part 3 GPU workflows.

The ABM continues to own feasible actions, geometry, workflow, patient state, and logging. Sparse decision evidence is exported only through explicit Part 3 flags. The vLLM backend is never reached by the default Part 1/2 pipeline; closed-loop use is available only through the dedicated Part 3 runner and broker.

## Deprecated Interpretation

Cooldowns alone are no longer the full memory story: Part 1/2 use cooldowns as bounded interaction constraints, while richer memory/reflection code exists only for optional Part 3 work. Part 2 is completed, not a proposed pilot.
