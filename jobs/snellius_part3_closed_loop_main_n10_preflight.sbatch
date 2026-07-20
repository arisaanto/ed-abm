#!/bin/bash
# CPU-only deployment gate for the 400-run paired Part 3 main study.

#SBATCH --job-name=abm_p3_main_pre
#SBATCH --partition=rome
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=8G
#SBATCH --time=00:30:00
#SBATCH --output=logs/abm_p3_main_pre_%j.out
#SBATCH --error=logs/abm_p3_main_pre_%j.err

set -euo pipefail

PROJECT_DIR="${PROJECT_DIR:-${HOME}/ABM/ABM}"
VENV_DIR="${VENV_DIR:-${HOME}/vllm-env}"
EVIDENCE_DIR="${EVIDENCE_DIR:-${HOME}/ABM_results/part3_evidence_logging_pilot}"
MANIFEST="${MANIFEST:-${PROJECT_DIR}/manifests/part3_closed_loop_main_n10.csv}"
SOURCE_LOCK="${SOURCE_LOCK:-${PROJECT_DIR}/manifests/part3_main_execution_source.sha256}"
OUT_JSON="${OUT_JSON:-${PROJECT_DIR}/logs/abm_p3_main_pre_${SLURM_JOB_ID}.json}"
RESUME_RESULTS_DIR="${RESUME_RESULTS_DIR:-}"
HF_HOME="${HF_HOME:-/scratch-shared/${USER}/huggingface}"
MODEL_ID="${MODEL_ID:-Qwen/Qwen3.6-35B-A3B}"
MODEL_CACHE_DIR="${HF_HOME}/hub/models--Qwen--Qwen3.6-35B-A3B"
MODEL_REVISION="${MODEL_REVISION:-}"

module purge
module load 2023
module load Python/3.11.3-GCCcore-12.3.0
source "${VENV_DIR}/bin/activate"

cd "${PROJECT_DIR}"
test -d "${EVIDENCE_DIR}"
test -f "${MANIFEST}"
test -f "${SOURCE_LOCK}"

# Refuse a partial or mixed-version deployment before importing project code.
sha256sum --check --strict "${SOURCE_LOCK}"

python3 scripts/check_part3_closed_loop_preflight.py \
  --episode-log "${EVIDENCE_DIR}" \
  --manifest "${MANIFEST}" \
  --study-design main \
  --expected-seed-count 10 \
  --guarded-path jobs/snellius_part3_qwen36_closed_loop_main_n10.sbatch \
  --guarded-path scripts/analyze_part3_closed_loop_main.py \
  --duration 8700 \
  --warmup-seconds 7200 \
  --max-model-decisions 5 \
  --max-model-decisions-per-agent 2 \
  --decision-window-seconds 300 \
  --max-model-decisions-per-window 1 \
  --out-json "${OUT_JSON}"

if [[ -n "${RESUME_RESULTS_DIR}" ]]; then
  if [[ ! -f "${RESUME_RESULTS_DIR}/main/manifest.csv" ]]; then
    echo "Resume audit requested but archived main manifest is missing." >&2
    exit 2
  fi
  if [[ -z "${MODEL_REVISION}" && -f "${MODEL_CACHE_DIR}/refs/main" ]]; then
    MODEL_REVISION="$(tr -d '[:space:]' < "${MODEL_CACHE_DIR}/refs/main")"
  fi
  if [[ -z "${MODEL_REVISION}" ]]; then
    echo "Set MODEL_REVISION or restore the cached model revision." >&2
    exit 2
  fi
  python3 scripts/run_part3_closed_loop_batch.py \
    --manifest "${MANIFEST}" \
    --output-root "${RESUME_RESULTS_DIR}/main" \
    --batch-name part3_closed_loop_main_n10 \
    --model "${MODEL_ID}" \
    --model-revision "${MODEL_REVISION}" \
    --tensor-parallel-size 4 \
    --max-model-len 8192 \
    --max-output-tokens 512 \
    --safetensors-load-strategy eager \
    --max-workers 60 \
    --inference-batch-wait-ms 75 \
    --resume \
    --resume-inventory-only

  python3 - "${OUT_JSON}" "${RESUME_RESULTS_DIR}/main/resume_inventory.json" <<'PY'
import json
from pathlib import Path
import sys

preflight_path = Path(sys.argv[1])
inventory_path = Path(sys.argv[2])
preflight = json.loads(preflight_path.read_text())
inventory = json.loads(inventory_path.read_text())
inventory["audit_pass"] = bool(
    inventory.get("inventory_pass") is True
    and inventory.get("inventory_only") is True
    and inventory.get("expected_run_count") == 400
    and inventory.get("recovered_complete_run_count", 0) > 0
    and inventory.get("pending_run_count", 0) > 0
    and inventory.get("recovered_complete_run_count", 0)
    + inventory.get("pending_run_count", 0)
    == 400
)
if not inventory["audit_pass"]:
    raise SystemExit(f"Resume inventory audit failed: {inventory}")
preflight["resume_inventory_audit"] = inventory
preflight_path.write_text(json.dumps(preflight, indent=2, sort_keys=True))
print(
    "Resume inventory passed: "
    f"{inventory['recovered_complete_run_count']} retained, "
    f"{inventory['pending_run_count']} pending"
)
PY
fi

echo "Passing main-study preflight: ${OUT_JSON}"
