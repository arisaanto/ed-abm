#!/bin/bash
# Cross 120 accepted decision contexts over five personas (600 decisions).
# Maximum estimate: 4 A100 x 0.5 h x 128 = 256 GPU SBU.

#SBATCH --job-name=abm_p3_common
#SBATCH --partition=gpu_a100
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --mem=460G
#SBATCH --gpus=4
#SBATCH --time=00:30:00
#SBATCH --output=logs/abm_p3_common_%j.out
#SBATCH --error=logs/abm_p3_common_%j.err

set -euo pipefail

PROJECT_DIR="${PROJECT_DIR:-${HOME}/ABM/ABM}"
VENV_DIR="${VENV_DIR:-${HOME}/vllm-env}"
INPUTS_DIR="${INPUTS_DIR:-${HOME}/ABM_results/part3_common_situation_audit_inputs}"
SCRATCH_DIR="${SCRATCH_DIR:-/scratch-shared/${USER}/abm/part3_common_situation_${SLURM_JOB_ID}}"
RESULTS_DIR="${RESULTS_DIR:-${HOME}/ABM_results/part3_common_situation_audit}"
HF_HOME="${HF_HOME:-/scratch-shared/${USER}/huggingface}"
PYTHON_MODULE="${PYTHON_MODULE:-Python/3.11.3-GCCcore-12.3.0}"
MODEL_ID="${MODEL_ID:-Qwen/Qwen3.6-35B-A3B}"
MODEL_CACHE_DIR="${HF_HOME}/hub/models--Qwen--Qwen3.6-35B-A3B"
MODEL_REVISION="${MODEL_REVISION:-995ad96eacd98c81ed38be0c5b274b04031597b0}"

module purge
module load 2023
module load "${PYTHON_MODULE}"
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

mkdir -p "${HF_HOME}" "${SCRATCH_DIR}" "${PROJECT_DIR}/logs"
if [[ ! -f "${INPUTS_DIR}/part3_common_situation_audit_preflight.json" ]]; then
  echo "Missing CPU-verified audit inputs" >&2
  exit 2
fi
if [[ -e "${RESULTS_DIR}" ]]; then
  echo "Refusing to overwrite existing result: ${RESULTS_DIR}" >&2
  exit 2
fi
if [[ ! -d "${MODEL_CACHE_DIR}/snapshots/${MODEL_REVISION}" ]]; then
  echo "Pinned model revision is absent from the offline cache" >&2
  exit 2
fi
cd "${PROJECT_DIR}"

sync_results() {
  mkdir -p "${RESULTS_DIR}"
  rsync -a "${SCRATCH_DIR}/" "${RESULTS_DIR}/"
}
trap sync_results EXIT

cp -a "${INPUTS_DIR}" "${SCRATCH_DIR}/inputs"
(cd "${PROJECT_DIR}" && sha256sum -c "${INPUTS_DIR}/source_sha256.txt")
PACKETS="${SCRATCH_DIR}/inputs/prompt_packets/part3_common_situation_audit_packets.jsonl"

python3 - <<'PY'
import torch
if not torch.cuda.is_available() or torch.cuda.device_count() != 4:
    raise SystemExit("GPU preflight failed: expected four A100 GPUs")
PY

python3 scripts/run/run_part3_vllm_offline.py \
  --input "${PACKETS}" \
  --output "${SCRATCH_DIR}/responses.jsonl" \
  --model "${MODEL_ID}" \
  --model-revision "${MODEL_REVISION}" \
  --tensor-parallel-size 4 \
  --max-model-len 8192 \
  --max-output-tokens 512 \
  --batch-size 32 \
  --semantic-retries 2 \
  --reject-fixtures \
  --summary-json "${SCRATCH_DIR}/inference_inventory.json"

python3 scripts/validation/verify_part3_vllm_responses.py \
  --input "${PACKETS}" \
  --responses "${SCRATCH_DIR}/responses.jsonl" \
  --reject-fixtures \
  --scientific-episode-count 120 \
  --out-json "${SCRATCH_DIR}/verification.json"

python3 scripts/analysis/analyze_part3_common_situation_audit.py \
  --packets "${PACKETS}" \
  --responses "${SCRATCH_DIR}/responses.jsonl" \
  --output-dir "${SCRATCH_DIR}/analysis"

python3 - "${SCRATCH_DIR}/verification.json" "${SCRATCH_DIR}/analysis/common_situation_audit_summary.json" <<'PY'
import json
from pathlib import Path
import sys
verification = json.loads(Path(sys.argv[1]).read_text())
analysis = json.loads(Path(sys.argv[2]).read_text())
if (
    verification.get("technical_verification_pass") is not True
    or verification.get("packet_count") != 600
    or verification.get("response_count") != 600
    or verification.get("missing_response_count") != 0
    or verification.get("unexpected_response_count") != 0
    or analysis.get("analysis_pass") is not True
    or analysis.get("context_count") != 120
):
    raise SystemExit("Common-situation audit verification failed")
print("Common-situation audit is technically complete")
print("Prespecified directions all passed:", analysis.get("all_prespecified_directions_pass"))
PY

sha256sum \
  "${PACKETS}" \
  "${SCRATCH_DIR}/responses.jsonl" \
  "${SCRATCH_DIR}/verification.json" \
  "${SCRATCH_DIR}/analysis/common_situation_audit_summary.json" \
  > "${SCRATCH_DIR}/result_sha256.txt"
