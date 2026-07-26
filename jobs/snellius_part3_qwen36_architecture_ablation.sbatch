#!/bin/bash
# Matched prompt-level persona x grounded-memory architecture ablation.
# 480 short categorical packets, one Qwen load, no ABM rerun.
# Maximum estimate: 4 A100 x 0.75 h x 128 = 384 GPU SBU.
# Expected observed cost is substantially lower because inference is short and batched.

#SBATCH --job-name=abm_p3_ablate
#SBATCH --partition=gpu_a100
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --mem=460G
#SBATCH --gpus=4
#SBATCH --time=00:45:00
#SBATCH --output=logs/abm_p3_ablate_%j.out
#SBATCH --error=logs/abm_p3_ablate_%j.err

set -euo pipefail

PROJECT_DIR="${PROJECT_DIR:-${HOME}/ABM/ABM}"
VENV_DIR="${VENV_DIR:-${HOME}/vllm-env}"
INPUTS_DIR="${INPUTS_DIR:-${HOME}/ABM_results/part3_architecture_ablation_inputs}"
SCRATCH_DIR="${SCRATCH_DIR:-/scratch-shared/${USER}/abm/part3_qwen36_architecture_ablation_${SLURM_JOB_ID}}"
RESULTS_DIR="${RESULTS_DIR:-${HOME}/ABM_results/part3_qwen36_architecture_ablation}"
HF_HOME="${HF_HOME:-/scratch-shared/${USER}/huggingface}"
PYTHON_MODULE="${PYTHON_MODULE:-Python/3.11.3-GCCcore-12.3.0}"
CUDA_MODULE="${CUDA_MODULE:-}"
MODEL_ID="${MODEL_ID:-Qwen/Qwen3.6-35B-A3B}"
MODEL_CACHE_DIR="${HF_HOME}/hub/models--Qwen--Qwen3.6-35B-A3B"
MODEL_REVISION="${MODEL_REVISION:-}"

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

sync_results() {
  rsync -a "${SCRATCH_DIR}/" "${RESULTS_DIR}/"
}
trap sync_results EXIT

if [[ ! -f "${INPUTS_DIR}/part3_architecture_ablation_preflight.json" ]]; then
  echo "Missing CPU-verified architecture-ablation inputs" >&2
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
PACKET_FILE="${SCRATCH_DIR}/inputs/prompt_packets/part3_architecture_ablation_packets.jsonl"

if [[ -z "${MODEL_REVISION}" && -f "${MODEL_CACHE_DIR}/refs/main" ]]; then
  MODEL_REVISION="$(tr -d '[:space:]' < "${MODEL_CACHE_DIR}/refs/main")"
fi
if [[ -z "${MODEL_REVISION}" ]]; then
  echo "Set MODEL_REVISION or restore the cached model revision" >&2
  exit 2
fi

python3 - "${SCRATCH_DIR}/inputs/part3_architecture_ablation_preflight.json" "${MODEL_REVISION}" <<'PY'
import json
from pathlib import Path
import sys

design = json.loads(Path(sys.argv[1]).read_text())
revision = sys.argv[2]
revisions = design.get("source_model_revision_counts") or {}
if (
    design.get("preflight_pass") is not True
    or design.get("packet_count") != 480
    or revisions != {revision: 120}
    or design.get("maximum_estimated_tokens", 99999) + 512 > 8192
):
    raise SystemExit(
        f"Ablation source revision/input mismatch: source={revisions}, run={revision}"
    )
print(f"Pinned architecture ablation to accepted main-study revision {revision}")
PY

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
  --max-output-tokens 512 \
  --batch-size 32 \
  --semantic-retries 2 \
  --reject-fixtures \
  --summary-json "${SCRATCH_DIR}/inference_summary.json"

python3 scripts/validation/verify_part3_vllm_responses.py \
  --input "${PACKET_FILE}" \
  --responses "${SCRATCH_DIR}/responses.jsonl" \
  --out-json "${SCRATCH_DIR}/verification.json" \
  --reject-fixtures

python3 - "${SCRATCH_DIR}/verification.json" <<'PY'
import json
from pathlib import Path
import sys

result = json.loads(Path(sys.argv[1]).read_text())
if (
    result.get("technical_verification_pass") is not True
    or result.get("architecture_ablation_checked") is not True
    or result.get("scientific_design_checked") is not False
    or result.get("packet_count") != 480
    or result.get("response_count") != 480
    or result.get("schema_or_grounding_error_count") != 0
    or result.get("missing_response_count") != 0
    or result.get("unexpected_response_count") != 0
):
    raise SystemExit("Architecture-ablation response verification failed")
print("Architecture-ablation responses passed strict technical verification")
PY

python3 scripts/analysis/analyze_part3_architecture_ablation.py \
  --packets "${PACKET_FILE}" \
  --responses "${SCRATCH_DIR}/responses.jsonl" \
  --output-dir "${SCRATCH_DIR}/analysis"

python3 - "${SCRATCH_DIR}/analysis/architecture_ablation_summary.json" <<'PY'
import json
from pathlib import Path
import sys

result = json.loads(Path(sys.argv[1]).read_text())
if (
    result.get("analysis_pass") is not True
    or result.get("matched_pair_count") != 120
    or result.get("reconstructed_cell_count") != 480
):
    raise SystemExit("Architecture-ablation analysis failed")
print("Architecture ablation complete: 120 matched four-cell comparisons")
PY

sha256sum \
  "${PACKET_FILE}" \
  "${SCRATCH_DIR}/responses.jsonl" \
  "${SCRATCH_DIR}/analysis/architecture_ablation_pair_effects.csv" \
  > "${SCRATCH_DIR}/result_sha256.txt"
