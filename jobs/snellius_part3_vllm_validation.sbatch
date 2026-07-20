#!/bin/bash
# Cheap environment/code-path validation using the same offline vLLM runner as
# the real pilot. Estimated maximum: 1 A100 x 0.5 h x 128 = 64 GPU SBU.
# Before submitting, confirm the current module names with `module avail python`
# and `module avail cuda`, then override PYTHON_MODULE/CUDA_MODULE if needed.

#SBATCH --job-name=abm_p3_vllm_check
#SBATCH --partition=gpu_a100
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gpus=1
#SBATCH --time=00:30:00
#SBATCH --output=logs/abm_p3_vllm_check_%j.out
#SBATCH --error=logs/abm_p3_vllm_check_%j.err

set -euo pipefail

PROJECT_DIR="${PROJECT_DIR:-${HOME}/ABM/ABM}"
VENV_DIR="${VENV_DIR:-${HOME}/vllm-env}"
SCRATCH_DIR="${SCRATCH_DIR:-/scratch-shared/${USER}/abm/part3_vllm_validation}"
RESULTS_DIR="${RESULTS_DIR:-${HOME}/ABM_results/part3_vllm_validation}"
HF_HOME="${HF_HOME:-/scratch-shared/${USER}/huggingface}"
PYTHON_MODULE="${PYTHON_MODULE:-Python/3.11.3-GCCcore-12.3.0}"
CUDA_MODULE="${CUDA_MODULE:-}"
PROMPT_DIR="${PROMPT_DIR:-${PROJECT_DIR}/outputs/part3/qwen_grounding_check/prompt_packets}"

module purge
module load 2023
module load "${PYTHON_MODULE}"
if [[ -n "${CUDA_MODULE}" ]]; then module load "${CUDA_MODULE}"; fi
source "${VENV_DIR}/bin/activate"

export HF_HOME
export TRANSFORMERS_CACHE="${HF_HOME}/hub"
export VLLM_WORKER_MULTIPROC_METHOD=spawn
# vLLM 0.25 defaults to FlashInfer sampling, which JIT-compiles kernels and
# requires a system CUDA toolkit/nvcc. Use vLLM's native sampler so the
# prebuilt PyTorch/vLLM CUDA runtime is sufficient for offline inference.
export VLLM_USE_FLASHINFER_SAMPLER=0
mkdir -p "${HF_HOME}" "${SCRATCH_DIR}" "${RESULTS_DIR}"
cd "${PROJECT_DIR}"

python3 - <<'PY'
import torch
import vllm

print(f"vLLM: {vllm.__version__}")
print(f"PyTorch: {torch.__version__}")
print(f"PyTorch CUDA runtime: {torch.version.cuda}")
print(f"GPU available: {torch.cuda.is_available()}")
if not torch.cuda.is_available():
    raise SystemExit("GPU preflight failed: PyTorch cannot access the allocated GPU.")
print(f"GPU: {torch.cuda.get_device_name(0)}")
PY

python3 scripts/run_part3_vllm_offline.py \
  --input "${PROMPT_DIR}" \
  --output "${SCRATCH_DIR}/responses.jsonl" \
  --model Qwen/Qwen2.5-0.5B-Instruct \
  --tensor-parallel-size 1 \
  --max-model-len 4096 \
  --max-output-tokens 256 \
  --batch-size 4 \
  --limit 8 \
  --reject-fixtures

python3 scripts/verify_part3_vllm_responses.py \
  --input "${PROMPT_DIR}" \
  --responses "${SCRATCH_DIR}/responses.jsonl" \
  --out-json "${SCRATCH_DIR}/verification.json" \
  --limit 8 \
  --reject-fixtures

rsync -a "${SCRATCH_DIR}/" "${RESULTS_DIR}/"
