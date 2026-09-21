#!/bin/bash
# Main evolving-persona study: 2 loads x 4 spatial conditions x 10 seeds x
# 5 balanced persona assignments = 400 complete runs.

#SBATCH --job-name=abm_p3_evo10
#SBATCH --partition=gpu_a100
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=48
#SBATCH --mem=460G
#SBATCH --gpus=4
#SBATCH --time=06:00:00
#SBATCH --output=logs/abm_p3_evo10_%j.out
#SBATCH --error=logs/abm_p3_evo10_%j.err

set -euo pipefail

PROJECT_DIR="${PROJECT_DIR:-${HOME}/ABM/ABM}"
VENV_DIR="${VENV_DIR:-${HOME}/vllm-env}"
HF_HOME="${HF_HOME:-/scratch-shared/${USER}/huggingface}"
MODEL_ID="${MODEL_ID:-Qwen/Qwen3.6-35B-A3B}"
MODEL_REVISION="${MODEL_REVISION:-995ad96eacd98c81ed38be0c5b274b04031597b0}"
MANIFEST="${MANIFEST:-${PROJECT_DIR}/manifests/part3_evolving_main_n10.csv}"
PREFLIGHT_JSON="${PREFLIGHT_JSON:?Set PREFLIGHT_JSON to the passing CPU preflight JSON}"
EVOLVING_PILOT_JSON="${EVOLVING_PILOT_JSON:-${HOME}/ABM_results/part3_evolving_pilot_n1_26262436/evolving_pilot_integrity.json}"
SCRATCH_DIR="${SCRATCH_DIR:-/scratch-shared/${USER}/abm/part3_evolving_main_n10_${SLURM_JOB_ID}}"
RESULTS_DIR="${RESULTS_DIR:-${HOME}/ABM_results/part3_evolving_main_n10}"
MAX_WORKERS="${MAX_WORKERS:-40}"

module purge
module load 2023
module load Python/3.11.3-GCCcore-12.3.0
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

mkdir -p "${SCRATCH_DIR}" "${RESULTS_DIR}"
cd "${PROJECT_DIR}"

python3 - "${PREFLIGHT_JSON}" "${EVOLVING_PILOT_JSON}" <<'PY'
import hashlib
import json
from pathlib import Path
import sys

preflight = json.loads(Path(sys.argv[1]).read_text())
pilot = json.loads(Path(sys.argv[2]).read_text())
manifest = preflight.get("study_manifest_audit") or {}
if preflight.get("preflight_pass") is not True or preflight.get("gpu_used") is not False:
    raise SystemExit("CPU preflight did not pass")
if (
    manifest.get("audit_pass") is not True
    or manifest.get("study_design") != "main"
    or manifest.get("row_count") != 400
    or manifest.get("expected_seed_count") != 10
    or manifest.get("evolving_state_enabled") != [True]
    or manifest.get("evolving_state_intervals_seconds") != [7200]
):
    raise SystemExit("The evolving-persona main manifest did not pass")
if pilot.get("verification_pass") is not True or pilot.get("run_count") != 40:
    raise SystemExit("The evolving-persona architecture pilot did not pass")
expected = preflight.get("deployed_source_sha256") or {}
root = Path.cwd().resolve()
mismatches = [
    relative
    for relative, digest in expected.items()
    if not (root / relative).is_file()
    or hashlib.sha256((root / relative).read_bytes()).hexdigest() != digest
]
if mismatches:
    raise SystemExit("Source changed after preflight: " + ", ".join(sorted(mismatches)))
PY

RESUME_ARGS=()
SYNC_DELETE=0
if [[ -f "${RESULTS_DIR}/main/manifest.csv" ]]; then
  rsync -a "${RESULTS_DIR}/" "${SCRATCH_DIR}/"
  RESUME_ARGS+=(--resume)
  SYNC_DELETE=1
elif find "${RESULTS_DIR}" -mindepth 1 -print -quit | grep -q .; then
  echo "Results directory is nonempty but has no resumable manifest: ${RESULTS_DIR}" >&2
  exit 2
fi

sync_results() {
  if [[ "${SYNC_DELETE}" == "1" ]]; then
    rsync -a --delete "${SCRATCH_DIR}/" "${RESULTS_DIR}/"
  else
    rsync -a "${SCRATCH_DIR}/" "${RESULTS_DIR}/"
  fi
}
trap sync_results EXIT

cp "${PREFLIGHT_JSON}" "${SCRATCH_DIR}/accepted_preflight.json"
cp "${EVOLVING_PILOT_JSON}" "${SCRATCH_DIR}/accepted_evolving_pilot.json"

python3 - <<'PY'
import torch
import vllm
print(f"vLLM: {vllm.__version__}")
print(f"PyTorch: {torch.__version__}")
print(f"GPU count: {torch.cuda.device_count()}")
if not torch.cuda.is_available() or torch.cuda.device_count() != 4:
    raise SystemExit("Expected four A100 GPUs")
PY

sha256sum --check --strict manifests/part3_evolving_main_execution_source.sha256

python3 scripts/run/run_part3_closed_loop_batch.py \
  --manifest "${MANIFEST}" \
  --output-root "${SCRATCH_DIR}/main" \
  --batch-name part3_evolving_main_n10 \
  --model "${MODEL_ID}" \
  --model-revision "${MODEL_REVISION}" \
  --tensor-parallel-size 4 \
  --max-model-len 8192 \
  --max-output-tokens 512 \
  --safetensors-load-strategy eager \
  --max-workers "${MAX_WORKERS}" \
  --inference-batch-wait-ms 75 \
  --state-ablation-sample-per-persona 100 \
  "${RESUME_ARGS[@]}"

python3 scripts/validation/verify_part3_evolving_pilot.py \
  --output-root "${SCRATCH_DIR}/main" \
  --manifest "${MANIFEST}" \
  --study-design main \
  --expected-seed-count 10 \
  --out-json "${SCRATCH_DIR}/evolving_main_integrity.json"

python3 scripts/analysis/analyze_part3_closed_loop_main.py \
  --output-root "${SCRATCH_DIR}/main" \
  --integrity-json "${SCRATCH_DIR}/evolving_main_integrity.json" \
  --output-dir "${SCRATCH_DIR}/analysis"

echo "Passing evolving-persona main study: ${RESULTS_DIR}"
