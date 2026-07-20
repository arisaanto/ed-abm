# Part 3 vLLM Backend Setup

## Scope

Part 3 uses Qwen/Qwen3.6-35B-A3B through vLLM's offline `LLM` interface on
Snellius. It does not run a persistent API server. The completed 400-run
closed-loop main study is frozen; the next GPU endpoint is post-run appraisal
inference over CPU-built evidence packets.

The cognitive layer is disabled by default. Parts 1 and 2 do not import vLLM,
load a model, or require a GPU environment.

## Separate environments

Core ABM environment:

```bash
python3 -m venv ~/abm-env
source ~/abm-env/bin/activate
python3 -m pip install --upgrade pip
python3 -m pip install -r ~/ABM/ABM/requirements.txt
```

Part 3 GPU environment:

```bash
python3 -m venv ~/vllm-env
source ~/vllm-env/bin/activate
python3 -m pip install --upgrade pip
python3 -m pip install -r ~/ABM/ABM/requirements-llm.txt
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

## Current execution order

1. Preserve the accepted closed-loop main result at
   `~/ABM_results/part3_closed_loop_main_n10/`.
2. Run the CPU-only all-seed appraisal packet preflight:

```bash
cd ~/ABM/ABM
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

5. Technical verification is necessary but not sufficient. Complete the
   prespecified manual grounding/naturalness review before interpreting survey
   or interview outputs or exporting the persona explorer data.

The optional architecture ablation is prepared with
`scripts/build_part3_ablation_packets.py`. It is a complete 2 x 2 comparison of
full versus neutral workplace orientation and retained versus removed grounded
memory. Building its packets is CPU-only; inference is deferred until the main
appraisal endpoint passes and remaining GPU budget is known.

## Cheap environment validation

`jobs/snellius_part3_vllm_validation.sbatch` exercises the same offline backend
with a small model before any new large-model environment is trusted. It is an
environment check, not a scientific experiment.

## Budget

The original 3,000 GPU-SBU allocation was a planning envelope, not a target to
spend. Main-study inference is complete. Reserve remaining GPU budget for the
single appraisal endpoint, a justified repair if verification fails, and only
then the optional 2 x 2 architecture ablation. Update estimates from observed
Snellius billing before submission; do not infer cost only from wall time.

All appraisal and ablation packet construction, contract tests, static
analysis, manual-review templates, and persona-explorer work are CPU/local
tasks and should not consume GPU SBUs.
