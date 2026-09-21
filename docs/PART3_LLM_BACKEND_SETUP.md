# Part 3 vLLM Backend Setup

## Scope

Part 3 uses Qwen/Qwen3.6-35B-A3B through vLLM's offline `LLM` interface on
Snellius. It does not run a persistent API server. The 400-run closed-loop main
study, post-run appraisals, and matched experience ablation are complete and
frozen.

The cognitive layer is disabled by default. Parts 1 and 2 do not import vLLM,
load a model, or require a GPU environment.

## Separate environments

Core ABM environment:

```bash
python3 -m venv .venv
source .venv/bin/activate
python3 -m pip install --upgrade pip
python3 -m pip install -r requirements.txt
```

Part 3 GPU environment:

```bash
python3 -m venv vllm-env
source vllm-env/bin/activate
python3 -m pip install --upgrade pip
python3 -m pip install -r requirements-llm.txt
```

Use separate virtual environments so vLLM can resolve a compatible PyTorch,
Transformers, Hugging Face Hub, safetensors, and tokenizer stack. Before first
installation, check current Snellius modules rather than copying stale names:

```bash
module avail python
module avail cuda
```

## Model cache

Keep the model outside the home quota:

```bash
export HF_HOME=/scratch-shared/$USER/huggingface
mkdir -p "$HF_HOME"
```

The scientific jobs set Hugging Face and Transformers offline mode after the
model is cached. Record the exact model revision in every result package.
Reasoning mode remains disabled: the endpoints require concise structured JSON,
and extra reasoning tokens would increase cost without becoming admissible
scientific evidence.

## Original Snellius execution record

The paths and commands below document the completed original run. They are
not a portable quick start and are not required to inspect the published
aggregate findings.

1. Preserve the accepted accumulated-experience main result at
   `~/ABM_results/part3_evolving_main_n10/`.
2. Run the CPU-only all-seed appraisal packet preflight:

```bash
cd /path/to/ed-abm
mkdir -p logs
sbatch jobs/snellius_part3_appraisal_preflight_n10.sbatch
```

3. Inspect `appraisal_packet_preflight.json`. It must report 800 packets from
   100 four-condition-paired units, all ten seeds per scenario/persona, 4/3/3
   role balance, and clean experience-evidence gates.
4. Only after that passes, run the appraisal endpoint:

```bash
sbatch jobs/snellius_part3_qwen36_appraisals_n10.sbatch
```

5. Complete technical verification and the prespecified grounding and
   naturalness review before interpreting survey or interview outputs or
   exporting the persona explorer data.

The reported experience check replayed 500 completed-run decisions with the
same persona and opportunity, once with and once without accumulated
experience. Its [published table](../outputs/findings/part3_cognitive_personas_n10/tables/appendix_tableA_experience_ablation.csv)
reports how often the choice changed. The earlier 2 x 2 factorial prompt
ablation described in [the development plan](PART3_PLAN.md) is historical and
is not a reported result.

## Budget

GPU allocations were treated as ceilings rather than spending targets. The
accepted result packages record actual model revisions, packet inventories,
and verification outputs; rerunning accepted inference is unnecessary.

All appraisal and ablation packet construction, contract tests, static
analysis, manual-review templates, and persona-explorer work are CPU/local
tasks and should not consume GPU SBUs.
