#!/bin/bash
# Tiny live closed-loop mechanics gate, not a scientific Part 3 experiment.
# Maximum estimate: 4 A100 x 0.75 h x 128 = 384 GPU SBU. Expected use is
# materially lower because model loading dominated the earlier 10-minute jobs.

#SBATCH --job-name=abm_p3_loop
#SBATCH --partition=gpu_a100
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --gpus=4
#SBATCH --time=00:45:00
#SBATCH --output=logs/abm_p3_loop_%j.out
#SBATCH --error=logs/abm_p3_loop_%j.err

set -euo pipefail

PROJECT_DIR="${PROJECT_DIR:-${HOME}/ABM/ABM}"
VENV_DIR="${VENV_DIR:-${HOME}/vllm-env}"
SCRATCH_DIR="${SCRATCH_DIR:-/scratch-shared/${USER}/abm/part3_qwen36_closed_loop_gate_${SLURM_JOB_ID}}"
RESULTS_DIR="${RESULTS_DIR:-${HOME}/ABM_results/part3_qwen36_closed_loop_gate}"
HF_HOME="${HF_HOME:-/scratch-shared/${USER}/huggingface}"
PYTHON_MODULE="${PYTHON_MODULE:-Python/3.11.3-GCCcore-12.3.0}"
CUDA_MODULE="${CUDA_MODULE:-}"
MODEL_ID="${MODEL_ID:-Qwen/Qwen3.6-35B-A3B}"
MODEL_CACHE_DIR="${HF_HOME}/hub/models--Qwen--Qwen3.6-35B-A3B"
MODEL_REVISION="${MODEL_REVISION:-}"
PREFLIGHT_JSON="${PREFLIGHT_JSON:-}"

module purge
module load 2023
module load "${PYTHON_MODULE}"
if [[ -n "${CUDA_MODULE}" ]]; then module load "${CUDA_MODULE}"; fi
source "${VENV_DIR}/bin/activate"

export HF_HOME
export VLLM_WORKER_MULTIPROC_METHOD=spawn
export VLLM_USE_FLASHINFER_SAMPLER=0
mkdir -p "${HF_HOME}" "${SCRATCH_DIR}" "${RESULTS_DIR}"
cd "${PROJECT_DIR}"

if [[ -z "${PREFLIGHT_JSON}" || ! -f "${PREFLIGHT_JSON}" ]]; then
  echo "Set PREFLIGHT_JSON to a passing no-GPU closed-loop preflight result." >&2
  exit 2
fi
python3 - "${PREFLIGHT_JSON}" <<'PY'
import hashlib
import json
from pathlib import Path
import sys

result = json.loads(Path(sys.argv[1]).read_text())
if result.get("preflight_pass") is not True or result.get("gpu_used") is not False:
    raise SystemExit("No-GPU closed-loop preflight has not passed")
if result.get("validated_output_token_ceiling") != 512:
    raise SystemExit("No-GPU preflight did not validate the 512-token contract")
if result.get("provider_call_count") != 5 or result.get("provider_errors"):
    raise SystemExit("No-GPU preflight did not complete five error-free provider calls")
closed_loop = result.get("closed_loop_verification", {})
if closed_loop.get("verification_pass") is not True:
    raise SystemExit("No-GPU closed-loop verification did not pass")
packet_audit = result.get("static_packet_audit", {})
if (
    packet_audit.get("audit_pass") is not True
    or packet_audit.get("episode_count") != 320
    or packet_audit.get("packet_variants_checked") != 3200
):
    raise SystemExit("Full retained-evidence packet audit did not pass")
expected_hashes = result.get("deployed_source_sha256")
if not isinstance(expected_hashes, dict) or not expected_hashes:
    raise SystemExit("No-GPU preflight is missing deployed source hashes")
project_dir = Path.cwd().resolve()
mismatches = []
for relative, expected in expected_hashes.items():
    path = project_dir / relative
    observed = hashlib.sha256(path.read_bytes()).hexdigest() if path.is_file() else None
    if observed != expected:
        mismatches.append(relative)
if mismatches:
    raise SystemExit(
        "Source changed after no-GPU preflight; rerun it before GPU submission: "
        + ", ".join(sorted(mismatches))
    )
print(f"Accepted no-GPU preflight: {sys.argv[1]}")
PY

if find "${SCRATCH_DIR}" -mindepth 1 -maxdepth 1 -print -quit | grep -q .; then
  echo "Scratch output is not empty: ${SCRATCH_DIR}" >&2
  exit 2
fi
if find "${RESULTS_DIR}" -mindepth 1 -maxdepth 1 -print -quit | grep -q .; then
  echo "Results output is not empty: ${RESULTS_DIR}" >&2
  exit 2
fi
if [[ -z "${MODEL_REVISION}" && -f "${MODEL_CACHE_DIR}/refs/main" ]]; then
  MODEL_REVISION="$(tr -d '[:space:]' < "${MODEL_CACHE_DIR}/refs/main")"
fi
if [[ -z "${MODEL_REVISION}" ]]; then
  echo "Set MODEL_REVISION or restore the cached Hugging Face revision." >&2
  exit 2
fi

# The live gate needs both the ABM scientific stack and vLLM in this one GPU
# environment. Install requirements-llm.txt into vllm-env once if this
# preflight reports a missing core package; it does not pin or replace torch.
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
    raise SystemExit("GPU preflight failed: expected four allocated GPUs")
PY

sha256sum \
  src/part3_closed_loop.py \
  src/interviews.py \
  src/simulation.py \
  src/vllm_backend.py \
  scripts/plan_part3_synthetic_study.py \
  scripts/check_part3_closed_loop_preflight.py \
  scripts/run_part3_closed_loop.py \
  scripts/run_single.py \
  scripts/verify_part3_closed_loop_gate.py \
  jobs/snellius_part3_qwen36_closed_loop_gate.sbatch \
  > "${SCRATCH_DIR}/source_sha256.txt"

RUN_DIR="${SCRATCH_DIR}/normal_load/both/seed_1"
mkdir -p "${RUN_DIR}"
python3 scripts/run_part3_closed_loop.py \
  --output-dir "${RUN_DIR}" \
  --model "${MODEL_ID}" \
  --model-revision "${MODEL_REVISION}" \
  --tensor-parallel-size 4 \
  --max-model-len 8192 \
  --max-output-tokens 512 \
  --scenario normal_load \
  --condition both \
  --seed 1 \
  --duration 9000 \
  --warmup-seconds 7200 \
  --assignment-round 1 \
  --max-model-decisions 5 \
  --max-model-decisions-per-agent 2

set +e
python3 scripts/verify_part3_closed_loop_gate.py \
  --run-dir "${RUN_DIR}" \
  --out-json "${SCRATCH_DIR}/verification.json"
VERIFY_STATUS=$?
set -e

rsync -a "${SCRATCH_DIR}/" "${RESULTS_DIR}/"
exit "${VERIFY_STATUS}"
