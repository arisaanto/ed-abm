#!/bin/bash
# One-seed closed-loop sizing pilot, not the final inferential Part 3 study.
# Hard ceiling: 4 A100 x 2 h x 128 = 1,024 GPU SBU. Actual charging follows
# elapsed time. Expected use is materially lower because Qwen loads once, 20
# independent ABMs run concurrently through a shared inference broker, and
# each eligible episode has the same preregistered 10% selection probability.
# The 300/100/100 call limits are fail-fast safety ceilings, not quotas.

#SBATCH --job-name=abm_p3_pilot
#SBATCH --partition=gpu_a100
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --gpus=4
#SBATCH --time=02:00:00
#SBATCH --output=logs/abm_p3_pilot_%j.out
#SBATCH --error=logs/abm_p3_pilot_%j.err

set -euo pipefail

PROJECT_DIR="${PROJECT_DIR:-${HOME}/ABM/ABM}"
VENV_DIR="${VENV_DIR:-${HOME}/vllm-env}"
SCRATCH_DIR="${SCRATCH_DIR:-/scratch-shared/${USER}/abm/part3_closed_loop_pilot_${SLURM_JOB_ID}}"
RESULTS_DIR="${RESULTS_DIR:-${HOME}/ABM_results/part3_closed_loop_pilot}"
HF_HOME="${HF_HOME:-/scratch-shared/${USER}/huggingface}"
PYTHON_MODULE="${PYTHON_MODULE:-Python/3.11.3-GCCcore-12.3.0}"
CUDA_MODULE="${CUDA_MODULE:-}"
MODEL_ID="${MODEL_ID:-Qwen/Qwen3.6-35B-A3B}"
MODEL_CACHE_DIR="${HF_HOME}/hub/models--Qwen--Qwen3.6-35B-A3B"
MODEL_REVISION="${MODEL_REVISION:-}"
PREFLIGHT_JSON="${PREFLIGHT_JSON:-}"
MANIFEST="${MANIFEST:-${PROJECT_DIR}/manifests/part3_closed_loop_pilot.csv}"

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
mkdir -p "${HF_HOME}" "${SCRATCH_DIR}" "${RESULTS_DIR}"
cd "${PROJECT_DIR}"

if find "${SCRATCH_DIR}" -mindepth 1 -maxdepth 1 -print -quit | grep -q .; then
  echo "Scratch output is not empty: ${SCRATCH_DIR}" >&2
  exit 2
fi
if find "${RESULTS_DIR}" -mindepth 1 -maxdepth 1 -print -quit | grep -q .; then
  echo "Results output is not empty: ${RESULTS_DIR}" >&2
  exit 2
fi

sync_results() {
  rsync -a "${SCRATCH_DIR}/" "${RESULTS_DIR}/"
}
trap sync_results EXIT

if [[ -z "${PREFLIGHT_JSON}" || ! -f "${PREFLIGHT_JSON}" ]]; then
  echo "Set PREFLIGHT_JSON to the passing pilot no-GPU preflight JSON." >&2
  exit 2
fi
python3 - "${PREFLIGHT_JSON}" <<'PY'
import hashlib
import json
from pathlib import Path
import sys

result = json.loads(Path(sys.argv[1]).read_text())
if result.get("preflight_pass") is not True or result.get("gpu_used") is not False:
    raise SystemExit("Pilot no-GPU preflight has not passed")
memory = result.get("grounded_memory_policy_audit") or {}
if (
    memory.get("audit_pass") is not True
    or memory.get("memory_policy") != "grounded_realized_interactions_v1"
    or memory.get("same_colleague_present") is not True
    or memory.get("same_patient_cross_colleague_present") is not True
    or memory.get("unsupported_persona_memory_modifiers") != {}
):
    raise SystemExit("Grounded-memory policy audit did not pass")
manifest = result.get("pilot_manifest_audit") or {}
if (
    manifest.get("audit_pass") is not True
    or manifest.get("row_count") != 20
    or manifest.get("model_decision_sample_rates") != [0.1]
    or manifest.get("safety_contracts") != [[300, 100, 100]]
):
    raise SystemExit("Pilot manifest audit did not pass for exactly 20 rows")
packet = result.get("static_packet_audit") or {}
if packet.get("audit_pass") is not True or packet.get("packet_variants_checked") != 3200:
    raise SystemExit("Full retained-evidence packet audit did not pass")
if result.get("decision_window_seconds") != 300:
    raise SystemExit("Temporal decision-window preflight was not exercised")
temporal = result.get("temporal_budget_audit") or {}
if (
    temporal.get("audit_pass") is not True
    or temporal.get("provider_call_count") != 5
    or len(temporal.get("provider_calls_by_window", {})) != 5
):
    raise SystemExit("Deterministic five-window temporal budget audit did not pass")
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
    raise SystemExit("Pilot preflight is missing source hashes")
root = Path.cwd().resolve()
mismatches = []
for relative, expected in expected_hashes.items():
    path = root / relative
    observed = hashlib.sha256(path.read_bytes()).hexdigest() if path.is_file() else None
    if observed != expected:
        mismatches.append(relative)
if mismatches:
    raise SystemExit(
        "Source changed after no-GPU preflight: " + ", ".join(sorted(mismatches))
    )
print(f"Accepted pilot no-GPU preflight: {sys.argv[1]}")
PY
cp "${PREFLIGHT_JSON}" "${SCRATCH_DIR}/accepted_preflight.json"

# Independent ABM workers are single-threaded. The parent process alone owns
# vLLM, avoiding CUDA duplication while the workers use the allocated CPUs.
export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1

if [[ -z "${MODEL_REVISION}" && -f "${MODEL_CACHE_DIR}/refs/main" ]]; then
  MODEL_REVISION="$(tr -d '[:space:]' < "${MODEL_CACHE_DIR}/refs/main")"
fi
if [[ -z "${MODEL_REVISION}" ]]; then
  echo "Set MODEL_REVISION or restore the cached model revision." >&2
  exit 2
fi

python3 - <<'PY'
import matplotlib
import networkx
import numpy
import pandas
import scipy
import shapely
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
  src/part3_closed_loop.py \
  src/personas.py \
  src/simulation.py \
  src/vllm_backend.py \
  scripts/check_part3_closed_loop_preflight.py \
  scripts/plan_part3_synthetic_study.py \
  scripts/run_part3_closed_loop.py \
  scripts/run_part3_closed_loop_batch.py \
  scripts/run_single.py \
  scripts/verify_part3_closed_loop_gate.py \
  scripts/verify_part3_closed_loop_pilot.py \
  manifests/part3_closed_loop_pilot.csv \
  jobs/snellius_part3_qwen36_closed_loop_pilot.sbatch \
  > "${SCRATCH_DIR}/source_sha256.txt"

python3 scripts/run_part3_closed_loop_batch.py \
  --manifest "${MANIFEST}" \
  --output-root "${SCRATCH_DIR}/pilot" \
  --model "${MODEL_ID}" \
  --model-revision "${MODEL_REVISION}" \
  --tensor-parallel-size 4 \
  --max-model-len 8192 \
  --max-output-tokens 512 \
  --safetensors-load-strategy eager \
  --max-workers 20 \
  --inference-batch-wait-ms 50

python3 scripts/verify_part3_closed_loop_pilot.py \
  --output-root "${SCRATCH_DIR}/pilot" \
  --manifest "${MANIFEST}" \
  --out-json "${SCRATCH_DIR}/integrity_check.json"

python3 - "${SCRATCH_DIR}/integrity_check.json" <<'PY'
import json
from pathlib import Path
import sys

result = json.loads(Path(sys.argv[1]).read_text())
required = (
    "verification_pass",
    "technical_integrity_pass",
    "pilot_design_complete",
    "temporal_coverage_pass",
    "sampling_coverage_pass",
    "persona_exposure_balance_pass",
    "grounded_memory_coverage_pass",
    "scientific_readiness_for_main",
)
failed = [name for name in required if result.get(name) is not True]
if failed:
    raise SystemExit("Closed-loop sizing gate failed: " + ", ".join(failed))
print("Closed-loop sizing gate passed all technical and readiness checks")
PY
