#!/bin/bash
# Questionnaire construct translation: 390 bundles, one Qwen load, no ABM rerun.
# Maximum estimate: 4 A100 x 0.75 h x 128 = 384 GPU SBU.

#SBATCH --job-name=abm_p3_q
#SBATCH --partition=gpu_a100
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --mem=460G
#SBATCH --gpus=4
#SBATCH --time=00:45:00
#SBATCH --output=logs/abm_p3_q_%j.out
#SBATCH --error=logs/abm_p3_q_%j.err

set -euo pipefail

PROJECT_DIR="${PROJECT_DIR:-${HOME}/ABM/ABM}"
VENV_DIR="${VENV_DIR:-${HOME}/vllm-env}"
INPUTS_DIR="${INPUTS_DIR:-${HOME}/ABM_results/part3_evolving_questionnaire_audit_inputs}"
SCRATCH_DIR="${SCRATCH_DIR:-/scratch-shared/${USER}/abm/part3_evolving_questionnaire_audit_${SLURM_JOB_ID}}"
RESULTS_DIR="${RESULTS_DIR:-${HOME}/ABM_results/part3_evolving_questionnaire_audit}"
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

mkdir -p "${HF_HOME}" "${SCRATCH_DIR}" "${RESULTS_DIR}"
cd "${PROJECT_DIR}"

sync_results() {
  rsync -a "${SCRATCH_DIR}/" "${RESULTS_DIR}/"
}
trap sync_results EXIT

if [[ ! -f "${INPUTS_DIR}/questionnaire_audit_preflight.json" ]]; then
  echo "Missing CPU-verified questionnaire inputs" >&2
  exit 2
fi
if find "${SCRATCH_DIR}" -mindepth 1 -maxdepth 1 -print -quit | grep -q .; then
  echo "Scratch output is not empty: ${SCRATCH_DIR}" >&2
  exit 2
fi
if find "${RESULTS_DIR}" -mindepth 1 -maxdepth 1 -print -quit | grep -q .; then
  echo "Results output is not empty: ${RESULTS_DIR}" >&2
  exit 2
fi

(cd "${PROJECT_DIR}" && sha256sum -c "${INPUTS_DIR}/source_sha256.txt")
cp -a "${INPUTS_DIR}" "${SCRATCH_DIR}/inputs"
PACKET_FILE="${SCRATCH_DIR}/inputs/prompt_packets/questionnaire_audit_prompt_packets.jsonl"

if [[ ! -d "${MODEL_CACHE_DIR}/snapshots/${MODEL_REVISION}" ]]; then
  echo "Pinned model revision is absent from the offline cache" >&2
  exit 2
fi

python3 - <<'PY'
import torch
if not torch.cuda.is_available() or torch.cuda.device_count() != 4:
    raise SystemExit("GPU preflight failed: expected four A100 GPUs")
PY

python3 scripts/run/run_part3_vllm_offline.py \
  --input "${PACKET_FILE}" \
  --output "${SCRATCH_DIR}/responses.jsonl" \
  --model "${MODEL_ID}" \
  --model-revision "${MODEL_REVISION}" \
  --tensor-parallel-size 4 \
  --max-model-len 8192 \
  --max-output-tokens 2048 \
  --batch-size 16 \
  --semantic-retries 2 \
  --reject-fixtures \
  --summary-json "${SCRATCH_DIR}/inference_inventory.json"

python3 scripts/validation/verify_part3_questionnaire_audit.py \
  --packets "${PACKET_FILE}" \
  --responses "${SCRATCH_DIR}/responses.jsonl" \
  --out-json "${SCRATCH_DIR}/verification.json"

python3 scripts/analysis/analyze_part3_questionnaire_audit.py \
  --packets "${PACKET_FILE}" \
  --responses "${SCRATCH_DIR}/responses.jsonl" \
  --verification "${SCRATCH_DIR}/verification.json" \
  --output-dir "${SCRATCH_DIR}/analysis"

python3 - "${SCRATCH_DIR}/verification.json" <<'PY'
import json
from pathlib import Path
import sys

result = json.loads(Path(sys.argv[1]).read_text())
if (
    result.get("technical_verification_pass") is not True
    or result.get("packet_count") != 390
    or result.get("response_count") != 390
    or result.get("missing_response_count") != 0
    or result.get("unexpected_response_count") != 0
):
    raise SystemExit("Questionnaire technical verification failed")
print("Questionnaire inference is technically complete")
print("Internal scientific gate:", result.get("scientific_gate_pass"))
PY

sha256sum \
  "${PACKET_FILE}" \
  "${SCRATCH_DIR}/responses.jsonl" \
  "${SCRATCH_DIR}/verification.json" \
  "${SCRATCH_DIR}/analysis/questionnaire_audit_analysis.json" \
  > "${SCRATCH_DIR}/result_sha256.txt"
