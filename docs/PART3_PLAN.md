# Part 3: Cognitive Orientations in a Spatially Constrained ED ABM

## Status

**The 400-run bounded closed-loop study is complete; LLM behavior remains
disabled by default.** Part 1 validation and Part 2 intervention results are
frozen. Part 3 adds a bounded cognitive interaction layer without changing
geometry, movement, clinical workflow, patient flow, arrival logic, or
proximity rules.

The 60-response Phase 3 construct test is complete. It established grounded,
categorically distinct persona-conditioned choices on matched evidence. It did
not estimate population effects and is not itself a closed-loop experiment.
The final causal contract excludes generated rationale, free topic text, and
self-reported confidence; those narrative fields showed avoidable unsupported
inference and are reserved for later post-run reflection.

The subsequent main experiment completed 400/400 paired runs across two
scenarios, four spatial conditions, ten seeds, and five balanced assignment
rounds. Technical integrity, persona-role exposure, common exogenous arrival
streams, temporal coverage, and sampling coverage passed. The next outstanding
phase is verified offline inference over 800 paired end-of-shift survey and
interview packets; until that job completes, qualitative appraisal findings are
not available.

## Research Contribution

Part 3 asks whether the same feasible ED situation and spatial intervention is
experienced and acted upon differently by distinct cognitive orientations.
The contribution is not that synthetic agents reproduce human testimony. It is
a controlled computational experiment in **persona-by-environment interaction**:
when role, workflow, space, and opportunity are held constant, does a bounded
cognitive policy produce interpretable differences in discretionary contact,
memory, appraisal, and reflection?

## Experimental Unit

The primary factor is a role-independent cognitive orientation. ABM role is a
blocking/control variable, not the persona definition. Across a complete
five-round assignment block, every staff id receives every persona exactly
once. The scientific study repeats complete blocks across paired seeds, so each
staff-persona combination receives multiple independent observations rather
than one anecdotal exposure. This prevents role composition from masquerading
as a persona effect while preserving stochastic replication.

Part 3 also uses a dedicated exogenous patient-arrival/profile random stream
keyed only by scenario and seed. Every condition and persona-assignment round
within a matched scenario-seed block therefore receives the same arrival
attempts and ESI profile. Endogenous workflow, movement, interaction, and
cognitive decisions remain free to diverge. Part 1 and Part 2 retain their
validated original random-number lifecycle.

The five preregistered orientations are:

1. **The Team Connector**: protects shared awareness and initiates concise coordination.
2. **The Focus Protector**: protects task continuity and filters low-value interruption.
3. **The Patient Advocate**: prioritizes accessibility, explanation, dignity, and continuity.
4. **The Vigilant Monitor**: prioritizes weak signals, escalation risk, and situational awareness.
5. **The Adaptive Generalist**: changes strategy with urgency, workload, and spatial context.

Domain-specific workplace priors are the only persona constructs supplied to
prompts. For the eventual character selector, code derives a coarse ordinal
OCEAN display from those priors. That one-way crosswalk is presentation-only:
it is not psychometric measurement, does not control behavior, and is never
used as a continuous explanatory variable.

## Ownership Boundary

The rule-based ABM always owns:

- geometry, routing, collision, movement, and visibility;
- patient state, ESI, beds, arrivals, and clinical workflow;
- feasible partners, physical proximity, and protected-care constraints;
- candidate contact opportunities and interaction logging;
- operational pressure and integrity gates.

The optional cognitive policy may only:

- choose `engage`, `defer`, or `decline` for an ABM-supplied feasible opportunity;
- choose a bounded reason and canonical topic family;
- retrieve sparse, source-linked memories of realized interactions;
- produce checkpoint reflections only after the causal simulation phase;
- answer structured end-of-shift appraisal and interview prompts from supplied evidence.

The causal in-simulation decision contract is deliberately categorical: action,
bounded reason, bounded topic family, and cited evidence only. Free-text topic
wording, rationale, and model-reported confidence do not alter mechanics and are
not trusted as memory. Causal memory contains only realized, logged interactions
with source event ids. It records recency, colleague, patient context when one
exists, topic family, operational reason, interaction context, zone, and whether
the staff member initiated or received the contact. Prior model choices, missed
opportunities, and generated reflections are excluded. Natural-language
reflection is generated later through a separate trace-grounded endpoint.

It may not invent movement, partners, patient facts, diagnoses, clinical
outcomes, proximity, or events. It must not run at every timestep.

## Topic Policy

The canonical empirical topic families remain the in-simulation comparison
layer. More specific natural-language subtopics may emerge only in the later
reflection/interview endpoint, where they must cite supporting trace evidence.
They are exploratory qualitative material, not new validated topic classes and
never causal inputs to the ABM.

Topic diversity is not a success criterion. The ABM supplies task- and
episode-constrained topic sets, so convergence on planning, handoff, or patient
status topics may be the correct outcome. Topic distributions are interpreted
as conversations only for `engage` decisions; defer/decline outputs retain a
bounded potential topic for decision auditing but do not represent speech.

## Quantitative Appraisal

Each persona rates seven 1-to-7 constructs from trace evidence:

- team awareness;
- interruption burden;
- task continuity;
- patient accessibility;
- privacy and control;
- spatial legibility;
- overall person-space fit.

Scores require supporting event ids, a short rationale, confidence, and an
uncertainty note. They are model outputs, not questionnaire responses from
human participants. Model-reported confidence is retained only as a diagnostic
of response behavior; it is not assumed to be calibrated and is not a primary
scientific outcome.

## Qualitative Reflection

The interview proceeds from open to focused questions:

1. overall experience of the layout;
2. evidence-backed critical moments;
3. coordination versus task-focus tradeoff;
4. most supportive spatial feature;
5. most difficult spatial feature;
6. one counterfactual design change and its tradeoff.

Responses should describe recurring experience and meaning, not recite an
itinerary. Analysis is **computational qualitative content analysis** using a
frozen codebook, evidence-grounding checks, contradiction checks, and selective
human inspection. It is not human inter-rater thematic analysis.

The causal main job does not generate these responses. It first preserves a
post-warmup, per-agent evidence record for every run: movement and zone/mode
dwell, mutually visible colleague exposure, staff-staff and patient-facing
interaction participation, unique partners, realized model contacts, missed
opportunities, and grounded-memory retrieval. Survey and interview packets are
built from the fixed completed traces in a separate post-run endpoint. This
separation prevents synthetic self-report from influencing the causal ABM and
allows the appraisal protocol to be revised or audited without rerunning the
400 trajectories.

Survey dimensions are not forced when the retained trace cannot support them.
Such a response is marked `insufficient_evidence` with a null score rather than
deriving a precise rating from the persona description alone. Unrateable
responses remain missing data; they are not midpoint-imputed or interpreted as
neutral experience.

### Episodic spatial experience representation

Each completed Part 3 run writes a sparse, source-linked representation for
every staff agent. It segments the continuous trajectory at modeled event
boundaries: place, mode, and task transitions; travel episodes; sustained
colleague visibility; realized contacts; missed opportunities; cognitive
decisions; and workflow events. It then summarizes these records as place
nodes, traversed links, and salient source-event ids. This preserves where work
was experienced, what occurred there, what transitions recurred, and which
moments can support a later appraisal without sending a timestep transcript to
Qwen.

The representation is deliberately named an **episodic spatial experience
representation**, not a human cognitive map. It contains simulated events and
spatial context; it does not model neural representation, subjective
atmosphere, emotion, or human memory. Event-boundary logging is inspired by
event-segmentation research, but the implemented boundaries are transparent
model state changes rather than a claim that agents perceive events as humans
do.

The files are:

- `part3_experience_events.jsonl`: immutable source-linked event records;
- `part3_spatial_experience.jsonl`: one place/transition map per staff-shift;
- `part3_agent_experience.jsonl`: aggregate behavioral exposure measures.

Generated survey answers, interview prose, latent needs, and design hypotheses
are never written back into these files and never affect causal decisions.

### Appraisal claim layers

Every interview answer separates four epistemic layers:

1. a **grounded pattern**, supported by cited simulated events;
2. a **persona-conditioned interpretation** of that pattern;
3. an optional **latent need**, explicitly labeled as conjecture;
4. an optional **design hypothesis**, paired with a plausible tradeoff.

Only the first layer is an event claim. The latter layers are synthetic design
probes, not observations, staff testimony, discovered preferences, or
validated psychological measurements. The public first-person answer may read
naturally, but process language (`ABM`, metrics, logs, evidence ids, or persona
labels) is forbidden. Automated verification checks schema, citations, and
obvious leakage; a prespecified human review remains mandatory because fluency
and credible interpretation cannot be established mechanically.

## Architecture Ablation

Four variants separate language, persona, and memory effects:

1. rule-based Part 2 reference;
2. generic LLM without persona or memory;
3. persona-conditioned LLM without longitudinal memory;
4. persona-conditioned LLM with memory and reflection.

The ablation is not scored mainly by empirical topic similarity. Primary
outcomes are:

- integrity and schema validity;
- evidence grounding and hallucination rate;
- persona fidelity and separability;
- behavior-belief alignment;
- temporal coherence;
- spatial sensitivity and persona-by-environment differentiation;
- run-to-run stability;
- system-level interaction consequences;
- compute cost per valid grounded response.

Empirical topic similarity is a secondary plausibility boundary: an LLM should
not improve semantic resemblance by violating feasibility or fabricating facts.

## Analysis Contract

The primary statistical object is the within-staff, paired persona/condition
contrast. Models should retain scenario, role, staff id, seed, and assignment
round. Report estimates and confidence intervals before p-values. Persona
effects that vanish after controlling role are not interpreted as cognitive
effects.

Qualitative and quantitative outputs remain separate data layers. Narrative
fluency cannot substitute for event evidence, and narrative text is never
parsed back into clinical workflow state.

## Phased Execution

### Phase 0: freeze and static validation

- validate five persona specifications and the balanced rotation;
- validate schemas and prompt constraints without model inference;
- confirm Parts 1/2 defaults and source hashes remain unchanged.

### Phase 1: logging-only instrumentation

- export sparse decision episodes, evidence ids, feasible actions, canonical
  reasons/topics, and pre/post memory state;
- do not let an LLM alter behavior yet;
- inspect opportunity counts and token estimates.

The first evidence pilot is deliberately limited to one paired seed across the
two scenarios and four spatial conditions: eight complete CPU ABM shifts. Full
shifts are necessary because candidate decisions arise from evolving workflow,
location, visibility, patient context, and ED pressure. This is not an LLM run.
Each shift retains a deterministic sample of at most 40 feasible opportunities
after the two-hour warmup. Sampling uses a SHA-256 priority reservoir and never
the simulation random generator. Pilot outputs must match the corresponding
frozen Part 2 summaries and interaction-event CSVs before they are used.

The ABM excludes protected active clinical work from opportunistic external
interruption. Part 3 therefore tests persona differences only at
ABM-approved, externally interruptible staff-contact boundaries. The Focus
Protector is tested through sensitivity to relevance, urgency, attention, and
spatial diversion cost, not by allowing an LLM to interrupt an active task.

Run this phase only with the explicit `--export-part3-episodes` flag. The
default Part 1/2 execution path does not write cognitive episodes. The logger
captures the same random draw used by the rule policy and adds no random calls,
so enabling export does not alter the reference decision sequence.

### Phase 2: tiny-model code-path validation

- run eight explicit schema fixtures through the offline vLLM path;
- require valid JSON, bounded choices, valid evidence ids, and no invented facts;
- fixtures are marked non-scientific and are never included in analysis.

### Phase 3: bounded Qwen go/no-go test

- choose exactly 12 construct-diverse real episodes from the 320 retained CPU
  evidence records and cross every episode with all five personas (60 model
  responses);
- cross the same episode with all five personas and balance patient-linked and
  system-level contexts across the eight scenario-condition cells;
- blind Qwen to the rule-policy action, random draw, ABM reason label,
  scenario/condition labels, and identifying source ids; retain them only for
  evaluator-side paired comparisons;
- explicitly identify whether a patient context or active task is present and
  define all model-facing numeric score scales;
- define `engage`, `defer`, and `decline` as value-versus-attention-cost choices
  rather than treating every feasible opportunity as automatically worthwhile;
- label staff workflow modes unambiguously and prevent staff return/wait states
  from being interpreted as patient disposition;
- remove patient-specific topics when no patient context is supplied and remove
  task-continuity as a reason when no active task is present;
- apply persona priors silently: rationales may cite evidence and tradeoffs but
  not persona titles or numeric prior values;
- keep Qwen thinking disabled;
- use deterministic, schema-constrained decoding for the first correctness
  pilot and record the exact model revision and canonical packet hashes;
- evaluate grounding, within-episode persona divergence, persona fidelity,
  latency, and token cost;
- require complete 12-by-5 matching, at least one evidence situation with
  action-level persona divergence, and no persona pair that is categorically
  identical across all 12 situations;
- stop if persona labels can be recovered only from stylistic slogans rather
  than distinct evidence-sensitive decisions.

For sampling coverage only, not model behavior, the go/no-go selector defines
low/high urgency as `<0.50`/`>=0.80`, low/high diversion cost as
`<0.33`/`>=0.67`, and low/high ED pressure as `<0.50`/`>=0.85`. These bins do
not alter ABM thresholds or values supplied to Qwen; they only prevent the
small test set from collapsing onto one narrow context.

This 12-episode set is a construct stress test, not the final inferential
sample. It answers whether grounded persona-conditioned action differences are
possible with the current architecture. It cannot estimate stable persona,
condition, or persona-by-condition effects. Those require the larger paired
closed-loop design only if this gate passes.

`verification_pass` is a technical schema/completeness gate. Closed-loop
progression additionally requires review of every grounding flag; the verifier
therefore reports `scientific_manual_review_required` separately.

### Phase 4: closed-loop pilot

- first pass the local deterministic mock-controller check in
  `scripts/check_part3_closed_loop.py`;
- before the accepted main execution, pass
  `jobs/snellius_part3_closed_loop_main_n10_preflight.sbatch`, which exercises
  the packet builder, provider adapter, controller, simulation hook, exports,
  resume inventory, and strict verifier without loading a model;
- bind the GPU gate to that passing preflight JSON; the GPU job refuses source
  hashes that differ from the CPU-tested deployment;
- activate sparse model decisions only at the existing route-supported,
  externally interruptible staff-contact boundary;
- cap model calls per run and per agent; after a cap or invalid response, use
  the already-drawn rule action rather than inventing a new action;
- apply only the validated categorical action, bounded reason, and canonical
  topic family to the simulation;
- build memory deterministically from evidence plus categorical choice; never
  parse generated prose into causal state;
- compare the four ablation variants on paired seeds only after a tiny
  closed-loop GPU mechanics gate passes;
- retain all Part 1/2 workflow, route, proximity, and coordinate hard gates.

The local controller check is plumbing validation, not a scientific result. It
must show controller-disabled trajectory equivalence, at least two exercised
action categories, a causal interaction carrying the selected canonical topic,
zero free-text causal inputs, and zero movement/contact integrity failures.
The Snellius CPU preflight must additionally record five accepted provider
decisions, no provider errors, valid memory-bearing blinded packets, workflow
`PASS`, and zero hard-gate events. It is also not a scientific result.
The first GPU closed-loop gate must remain deliberately small and exists to
measure live call frequency, latency, fallback behavior, and integrity before
the paired pilot is sized. Another standalone 60-packet run is not required.

The first gate runs one normal-load `both` trajectory, leaves the controller
dormant through the two-hour warmup, and permits at most five Qwen calls during
the following 30 minutes. This is enough to test causal plumbing without
estimating persona or intervention effects. Qwen's categorical action changes
whether the opportunity is pursued; its bounded reason and canonical topic are
carried into any realized interaction and the resulting structured memory.
Generated prose is excluded because the validated causal contract contains no
prose field. What agents choose to discuss can therefore affect later choices
through remembered topic/reason/action categories, while unbounded wording
cannot silently become a new model-state variable.

The live mechanics gate has now passed with the pinned Qwen revision: five
validated model calls, both engage and non-engage outcomes, deterministic
memory reuse, workflow `PASS`, and zero route/contact/coordinate hard gates.
The next step is a **one-seed sizing pilot**, not the final experiment:

- two scenarios;
- the endpoint contrast `baseline` versus `both`;
- all five balanced persona-assignment rounds;
- 20 complete 12-hour runs as independent concurrent ABM workers;
- one parent-owned vLLM engine and synchronous inference broker shared by all
  workers, so Qwen still loads exactly once;
- deterministic 10% sampling of every eligible post-warmup decision episode,
  using a condition-invariant matched-opportunity hash that does not consume
  simulation randomness;
- fail-fast safety ceilings of 300 calls per run, 100 per agent, and 100 per
  two-hour window. Any ceiling hit invalidates the sizing run rather than
  silently changing its sampling fraction.

Every staff member receives every cognitive persona exactly once within each
scenario-condition cell. Hash-threshold sampling gives every eligible episode
the same selection probability and avoids the first-come bias found in the
canceled fixed-40 run. This pilot estimates event coverage, integrity, runtime,
and SBU cost; one seed is not sufficient for final persona or
persona-by-condition inference.

Parallelism does not make the policy asynchronous within a simulation. A
worker blocks at each cognitive decision until the shared broker returns that
packet's schema-constrained result. Independent runs can advance concurrently,
and requests that become ready together can be micro-batched. This preserves
each trajectory's causal order while avoiding 20 serial 12-hour ABM runs and
20 separate model loads. The retained batch metadata must show all 20 workers,
overlapping execution, one engine load, and equality between broker packets
and verified provider calls.

### Phase 5: bounded main study

The verified grounded-memory one-seed sizing pilot completed 20 runs with one
model load, 10,094 eligible decisions, 1,030 sampled Qwen decisions, and all
technical, temporal, sampling, persona-balance, and grounded-memory gates
passing. Its one-seed persona contrasts are diagnostics only. The main study
is therefore fixed at:

- two scenarios (`normal_load` and `high_load_high_acuity`);
- all four spatial conditions (`baseline`, `cockpit_only`, `nursta_only`, and
  `both`);
- ten paired seeds;
- all five persona-to-role assignment rotations per cell;
- 400 complete runs and deterministic 10% model sampling;
- one parent-owned Qwen engine shared by up to 60 concurrent ABM workers.

Ten seeds are the bounded scientific design, not a reduced version of an n=100
plan. One Part 3 seed already expands to 40 complete trajectories (two
scenarios x four conditions x five rotations). An n=100 design would require
4,000 GPU-coupled trajectories and would spend scarce compute on precision far
beyond what five designed synthetic orientations can justify. The seed is the
inferential unit; rotations are averaged within seed and individual Qwen
decisions are never treated as independent replicates.

Primary analyses use paired condition deltas, the SD and SE of paired deltas,
Student-t 95% confidence intervals, and exact two-sided sign-flip tests. The
main interpretation emphasizes effect sizes and intervals rather than a binary
p-value threshold. System outcomes and persona engagement policies are
analyzed separately.

The sizing pilot's extreme Team Connector bed-flow contrast (100% engagement
under Baseline and approximately 3% under Both) was traced partly to an
inadequate rolling decision-history representation. A fixed-packet diagnostic
showed that removing it changed Both-scenario bed-flow engagement by roughly
29--37 percentage points. The main study is therefore blocked from using that
policy.

The replacement policy follows a bounded episodic-memory contract. It retrieves
at most five realized interactions from the preceding two hours, and only when
there is an explicit match to the current colleague, patient context,
operational reason, interaction context, or very recent zone. Same-patient
memory may come from a different colleague, permitting a constrained form of
information relay without inventing clinical content. Retrieval uses audited
recency, relevance, importance, and preregistered persona salience; every record
retains its source interaction id and retrieval trace. A revised 20-run pilot
exercised this policy in every endpoint scenario-condition cell. It produced
4,378 grounded-memory exposures, no provider or workflow failures, and no
future, unsupported, or cross-run memory. The main-study export now also
writes an evaluator-only source-event ledger for every retrieved memory,
including pre-warmup events, so causal memory can be independently traced
without exposing that ledger to the model.

The main analysis stratifies decisions by staff role, ABM reason,
diversion-cost band, and grounded memory context (none, same colleague, same
patient, or other matched experience) before interpreting a
persona-by-environment effect. Persona contrasts are reported both for the
observed opportunity ecology and under a fixed 1:5:3 coordination-nurse,
nurse, and doctor composition. Each designed orientation also has a
directional construct-coherence diagnostic. The original one-seed contrast is
not tuned away, accepted as a result, or used as an integrity gate.

The main package requires exactly nine `part3_agent_experience.jsonl` rows per
run (3,600 total). These records add a second, behaviorally observed experience
layer: movement per agent-hour, visibility exposure, interaction participation,
partner breadth, model decision-to-realized-contact yield, and grounded-memory
retrieval. They are aggregated within persona and seed before paired condition
contrasts; individual agent-shifts are not treated as independent replicates.

The package also requires one source-event ledger and one episodic spatial
experience map for every staff-shift. After the 400-run causal study passes all
integrity gates, the post-run appraisal builder retains every seed for every
scenario-persona block. It chooses one complete four-condition-paired staff
shift per seed while balancing roles 4/3/3 across the ten seeds. A paired unit
fixes scenario, seed, assignment round, persona, role, and staff id. This yields
100 paired units, 400 agent-shifts, 400 survey bundles, and 400 interview
bundles. Sampling is deterministic and condition-paired; the agent-shifts are
not presented as independent human participants.

Survey/interview inference is a separate job after main-study acceptance. It
loads Qwen once, uses thinking-disabled structured decoding, and cannot alter
completed trajectories. One full interview per scenario-condition-persona cell
(40 bundles) is selected deterministically for mandatory review of grounding,
naturalness, conjecture labels, and design tradeoffs. Automated theme discovery
is not performed before that review.

Phase 5 does not include a broad architecture grid. A no-memory ablation is
only justified after the paired main result if grounded-memory context
materially changes interpretation. This preserves the remaining
GPU budget for an evidence-driven follow-up rather than a speculative rerun.

### Prepared component ablations (not executed)

Two matched, post-run packet ablations are prepared for an evidence-driven
follow-up without rerunning the ABM:

- **Neutral orientation:** replace the five designed decision priors with a
  common 0.5 profile while retaining the exact evidence and originally
  retrieved grounded memories.
- **No memory:** retain the assigned persona and exact decision evidence while
  removing the model-visible episodic-memory context.

`scripts/build_part3_ablation_packets.py` selects one memory-bearing model
opportunity per scenario-condition-persona-role stratum from the accepted main
study. The resulting 120 matched source opportunities produce 240 blinded
packets, while the original full-persona/full-memory Qwen response remains
evaluator-side as the reference. The CPU preflight requires complete strata,
exact variant pairing, no memory leakage, a genuinely neutral profile, and no
evaluator metadata in the model-visible packet.

These are prompt-level component substitutions, not new closed-loop
trajectories. They can test whether orientation and memory alter decisions on
fixed observed opportunities. They cannot estimate how a no-memory or neutral
policy would recursively change the later opportunity ecology. Execution is
therefore optional and should occur only if the appraisal and main-study
interpretation justify its GPU cost.

## Failure Criteria

Pause Part 3 if any of the following occurs:

- an LLM changes movement/workflow or selects an infeasible partner/action;
- unsupported clinical facts or patient outcomes appear;
- evidence citations are absent or invalid;
- persona effects are mostly role effects or prompt catchphrases;
- the balanced sizing pilot shows no action-level persona signal or no
  defensible persona-by-condition contrast large enough to justify the main
  study under the remaining SBU budget;
- the same evidence produces unstable decisions beyond the preregistered tolerance;
- structured response validity is too low for reproducible analysis;
- projected GPU use exceeds the budget without a defensible scientific gain.

## Validity Boundary

Part 3 can support claims about a constrained synthetic cognitive experiment
and design-sensitive computational personas. It cannot establish real staff
preferences, population personality distributions, psychological validity,
clinical outcomes, or replacement of participatory/post-occupancy research.

The eventual character selector is a supplementary communication interface.
Static figures and conventional tables remain the primary paper outputs.
