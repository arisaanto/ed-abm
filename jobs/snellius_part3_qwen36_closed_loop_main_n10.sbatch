#!/bin/bash
# Paired Part 3 main study: 2 scenarios x 4 conditions x 10 seeds x
# 5 balanced persona rotations = 400 complete runs.
#
# Four A100 GPUs cost 512 SBU per wall-clock hour. The one-seed sizing pilot
# used ~170 SBU in 19.9 minutes with 20 workers. With 64 workers and one model
# load, the main-study estimate is roughly 1,000-1,300 SBU. The 2 h 45 min
# limit is a hard ceiling of 1,408 SBU, not an expected charge.

#SBATCH --job-name=abm_p3_main10
#SBATCH --partition=gpu_a100
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=64
#SBATCH --mem=460G
#SBATCH --gpus=4
#SBATCH --time=02:45:00
#SBATCH --output=logs/abm_p3_main10_%j.out
#SBATCH --error=logs/abm_p3_main10_%j.err

set -euo pipefail

PROJECT_DIR="${PROJECT_DIR:-${HOME}/ABM/ABM}"
VENV_DIR="${VENV_DIR:-${HOME}/vllm-env}"
SCRATCH_DIR="${SCRATCH_DIR:-/scratch-shared/${USER}/abm/part3_closed_loop_main_n10_${SLURM_JOB_ID}}"
RESULTS_DIR="${RESULTS_DIR:-${HOME}/ABM_results/part3_closed_loop_main_n10}"
HF_HOME="${HF_HOME:-/scratch-shared/${USER}/huggingface}"
PYTHON_MODULE="${PYTHON_MODULE:-Python/3.11.3-GCCcore-12.3.0}"
CUDA_MODULE="${CUDA_MODULE:-}"
MODEL_ID="${MODEL_ID:-Qwen/Qwen3.6-35B-A3B}"
MODEL_CACHE_DIR="${HF_HOME}/hub/models--Qwen--Qwen3.6-35B-A3B"
MODEL_REVISION="${MODEL_REVISION:-}"
PREFLIGHT_JSON="${PREFLIGHT_JSON:-}"
MEMORY_PILOT_JSON="${MEMORY_PILOT_JSON:-}"
MANIFEST="${MANIFEST:-${PROJECT_DIR}/manifests/part3_closed_loop_main_n10.csv}"
# Sixty workers fill the same seven scheduling waves as 64 while leaving
# twelve node CPUs for the parent broker and four vLLM worker processes.
MAX_WORKERS="${MAX_WORKERS:-60}"
RESUME="${RESUME:-0}"

module purge
module load 2023
module load "${PYTHON_MODULE}"
if [[ -n "${CUDA_MODULE}" ]]; then module load "${CUDA_MODULE}"; fi
source "${VENV_DIR}/bin/activate"

export HF_HOME
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export VLLM_WORKER_MULTIPROC_METHOD=spawn
export VLLM_USE_FLASHINFER_SAMPLER=0
export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1

mkdir -p "${HF_HOME}" "${SCRATCH_DIR}" "${RESULTS_DIR}"
cd "${PROJECT_DIR}"

SYNC_READY=0
sync_results() {
  if [[ "${SYNC_READY}" == "1" ]]; then
    if [[ "${RESUME}" == "1" ]]; then
      rsync -a --delete "${SCRATCH_DIR}/" "${RESULTS_DIR}/"
    else
      rsync -a "${SCRATCH_DIR}/" "${RESULTS_DIR}/"
    fi
  fi
}
trap sync_results EXIT

if [[ -z "${PREFLIGHT_JSON}" || ! -f "${PREFLIGHT_JSON}" ]]; then
  echo "Set PREFLIGHT_JSON to the passing main-study CPU preflight JSON." >&2
  exit 2
fi
if [[ -z "${MEMORY_PILOT_JSON}" || ! -f "${MEMORY_PILOT_JSON}" ]]; then
  echo "Set MEMORY_PILOT_JSON to the passing revised-memory pilot integrity JSON." >&2
  exit 2
fi

python3 - "${PREFLIGHT_JSON}" "${RESUME}" <<'PY'
import hashlib
import json
from pathlib import Path
import sys

result = json.loads(Path(sys.argv[1]).read_text())
resume = sys.argv[2] == "1"
if result.get("preflight_pass") is not True or result.get("gpu_used") is not False:
    raise SystemExit("Main-study no-GPU preflight has not passed")
memory = result.get("grounded_memory_policy_audit") or {}
if (
    memory.get("audit_pass") is not True
    or memory.get("memory_policy") != "grounded_realized_interactions_v1"
    or memory.get("same_colleague_present") is not True
    or memory.get("same_patient_cross_colleague_present") is not True
    or memory.get("unsupported_persona_memory_modifiers") != {}
):
    raise SystemExit("Grounded-memory policy audit did not pass")
manifest = result.get("study_manifest_audit") or {}
if (
    manifest.get("audit_pass") is not True
    or manifest.get("study_design") != "main"
    or manifest.get("row_count") != 400
    or manifest.get("expected_seed_count") != 10
    or manifest.get("expected_conditions")
    != ["baseline", "both", "cockpit_only", "nursta_only"]
    or manifest.get("model_decision_sample_rates") != [0.1]
    or manifest.get("safety_contracts") != [[300, 100, 100]]
):
    raise SystemExit("Main manifest audit did not pass for exactly 400 rows")
packet = result.get("static_packet_audit") or {}
if packet.get("audit_pass") is not True or packet.get("packet_variants_checked") != 3200:
    raise SystemExit("Full retained-evidence packet audit did not pass")
appraisal = result.get("appraisal_packet_audit") or {}
if (
    appraisal.get("audit_pass") is not True
    or appraisal.get("packet_count") != 18
    or appraisal.get("claim_layer_contract")
    != "grounded_pattern_interpretation_conjecture_v1"
    or appraisal.get("gpu_used") is not False
    or appraisal.get("response_verification_pass") is not True
    or appraisal.get("human_review_required") is not True
    or appraisal.get("analysis_export_pass") is not True
):
    raise SystemExit("Post-run appraisal packet audit did not pass")
sampling = result.get("sampling_policy_audit") or {}
if (
    sampling.get("audit_pass") is not True
    or sampling.get("eligible_episode_count") != 1000
    or sampling.get("repeat_identical") is not True
    or sampling.get("condition_matched") is not True
    or sampling.get("assignment_rounds_independent") is not True
):
    raise SystemExit("Deterministic proportional-sampling audit did not pass")
parallel = result.get("parallel_broker_audit") or {}
if (
    parallel.get("audit_pass") is not True
    or parallel.get("completed_run_count") != 2
    or parallel.get("maximum_active_workers") != 2
):
    raise SystemExit("Parallel shared-inference broker audit did not pass")
expected_hashes = result.get("deployed_source_sha256")
if not isinstance(expected_hashes, dict) or not expected_hashes:
    raise SystemExit("Main preflight is missing source hashes")
root = Path.cwd().resolve()
mismatches = []
for relative, expected in expected_hashes.items():
    path = root / relative
    observed = hashlib.sha256(path.read_bytes()).hexdigest() if path.is_file() else None
    if observed != expected:
        mismatches.append(relative)
if mismatches:
    raise SystemExit(
        "Source changed after main-study preflight: "
        + ", ".join(sorted(mismatches))
    )
if resume:
    inventory = result.get("resume_inventory_audit") or {}
    if (
        inventory.get("audit_pass") is not True
        or inventory.get("inventory_only") is not True
        or inventory.get("expected_run_count") != 400
        or inventory.get("recovered_complete_run_count", 0) <= 0
        or inventory.get("pending_run_count", 0) <= 0
        or inventory.get("recovered_complete_run_count", 0)
        + inventory.get("pending_run_count", 0)
        != 400
    ):
        raise SystemExit("CPU resume-inventory gate did not pass")
print(f"Accepted main-study no-GPU preflight: {sys.argv[1]}")
PY

python3 - "${MEMORY_PILOT_JSON}" <<'PY'
import json
from pathlib import Path
import sys

result = json.loads(Path(sys.argv[1]).read_text())
if (
    result.get("verification_pass") is not True
    or result.get("technical_integrity_pass") is not True
    or result.get("pilot_design_complete") is not True
    or result.get("scientific_readiness_for_main") is not True
    or result.get("grounded_memory_coverage_pass") is not True
    or result.get("observed_summary_count") != 20
    or result.get("memory_policy") != "grounded_realized_interactions_v1"
    or result.get("decision_history_used_as_memory") is not False
    or result.get("generated_reflection_used_as_causal_input") is not False
):
    raise SystemExit(
        "Revised-memory closed-loop pilot has not cleared the main study"
    )
print("Revised grounded-memory pilot gate passed")
PY

if [[ "${RESUME}" == "1" ]]; then
  if [[ ! -f "${RESULTS_DIR}/main/manifest.csv" ]]; then
    echo "Resume requested but no prior main-study output was found." >&2
    exit 2
  fi
  if find "${SCRATCH_DIR}" -mindepth 1 -print -quit | grep -q .; then
    echo "Resume scratch directory must begin empty: ${SCRATCH_DIR}" >&2
    exit 2
  fi
  rsync -a "${RESULTS_DIR}/" "${SCRATCH_DIR}/"
  rm -rf \
    "${SCRATCH_DIR}/analysis" \
    "${SCRATCH_DIR}/integrity_check.json"
else
  if find "${SCRATCH_DIR}" -mindepth 1 -print -quit | grep -q .; then
    echo "Scratch output is not empty: ${SCRATCH_DIR}" >&2
    exit 2
  fi
  if find "${RESULTS_DIR}" -mindepth 1 -print -quit | grep -q .; then
    echo "Results output is not empty: ${RESULTS_DIR}" >&2
    exit 2
  fi
fi
cp "${PREFLIGHT_JSON}" "${SCRATCH_DIR}/accepted_preflight.json"
cp "${MEMORY_PILOT_JSON}" "${SCRATCH_DIR}/accepted_memory_pilot.json"
SYNC_READY=1
if [[ -z "${MODEL_REVISION}" && -f "${MODEL_CACHE_DIR}/refs/main" ]]; then
  MODEL_REVISION="$(tr -d '[:space:]' < "${MODEL_CACHE_DIR}/refs/main")"
fi
if [[ -z "${MODEL_REVISION}" ]]; then
  echo "Set MODEL_REVISION or restore the cached model revision." >&2
  exit 2
fi

python3 - <<'PY'
import torch
import vllm

print(f"vLLM: {vllm.__version__}")
print(f"PyTorch: {torch.__version__}")
print(f"GPU count: {torch.cuda.device_count()}")
if not torch.cuda.is_available() or torch.cuda.device_count() != 4:
    raise SystemExit("GPU preflight failed: expected four A100 GPUs")
PY

sha256sum \
  src/interaction.py \
  src/interviews.py \
  src/part3_closed_loop.py \
  src/personas.py \
  src/simulation.py \
  src/vllm_backend.py \
  scripts/analysis/analyze_part3_closed_loop_main.py \
  scripts/build/build_part3_appraisal_packets.py \
  scripts/validation/check_part3_closed_loop_preflight.py \
  scripts/build/plan_part3_synthetic_study.py \
  scripts/run/run_part3_closed_loop.py \
  scripts/run/run_part3_closed_loop_batch.py \
  scripts/run/run_single.py \
  scripts/validation/verify_part3_closed_loop_gate.py \
  scripts/validation/verify_part3_closed_loop_pilot.py \
  manifests/part3_closed_loop_main_n10.csv \
  jobs/snellius_part3_qwen36_closed_loop_main_n10.sbatch \
  > "${SCRATCH_DIR}/source_sha256.txt"

RUNNER_RESUME_ARGS=()
if [[ "${RESUME}" == "1" ]]; then
  RUNNER_RESUME_ARGS+=(--resume)
fi

python3 scripts/run/run_part3_closed_loop_batch.py \
  --manifest "${MANIFEST}" \
  --output-root "${SCRATCH_DIR}/main" \
  --batch-name part3_closed_loop_main_n10 \
  --model "${MODEL_ID}" \
  --model-revision "${MODEL_REVISION}" \
  --tensor-parallel-size 4 \
  --max-model-len 8192 \
  --max-output-tokens 512 \
  --safetensors-load-strategy eager \
  --max-workers "${MAX_WORKERS}" \
  --inference-batch-wait-ms 75 \
  "${RUNNER_RESUME_ARGS[@]}"

python3 scripts/validation/verify_part3_closed_loop_pilot.py \
  --output-root "${SCRATCH_DIR}/main" \
  --manifest "${MANIFEST}" \
  --study-design main \
  --expected-seed-count 10 \
  --out-json "${SCRATCH_DIR}/integrity_check.json"

python3 - "${SCRATCH_DIR}/integrity_check.json" <<'PY'
import json
from pathlib import Path
import sys

result = json.loads(Path(sys.argv[1]).read_text())
required = (
    "verification_pass",
    "technical_integrity_pass",
    "main_design_complete",
    "study_design_complete",
    "temporal_coverage_pass",
    "sampling_coverage_pass",
    "persona_exposure_balance_pass",
    "agent_experience_export_pass",
    "experience_event_export_pass",
    "spatial_experience_map_export_pass",
    "scientific_readiness_for_main",
    "scientific_result",
)
failed = [name for name in required if result.get(name) is not True]
if failed:
    raise SystemExit("Closed-loop main study failed: " + ", ".join(failed))
print("Closed-loop main study passed all integrity and design checks")
PY

python3 scripts/analysis/analyze_part3_closed_loop_main.py \
  --output-root "${SCRATCH_DIR}/main" \
  --integrity-json "${SCRATCH_DIR}/integrity_check.json" \
  --output-dir "${SCRATCH_DIR}/analysis"
