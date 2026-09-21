#!/bin/bash
# CPU-only deployment gate for the 400-run evolving-persona main study.

#SBATCH --job-name=abm_p3_evo10_pre
#SBATCH --partition=rome
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=12G
#SBATCH --time=00:30:00
#SBATCH --output=logs/abm_p3_evo10_pre_%j.out
#SBATCH --error=logs/abm_p3_evo10_pre_%j.err

set -euo pipefail

PROJECT_DIR="${PROJECT_DIR:-${HOME}/ABM/ABM}"
VENV_DIR="${VENV_DIR:-${HOME}/vllm-env}"
EVIDENCE_DIR="${EVIDENCE_DIR:-${HOME}/ABM_results/part3_evidence_logging_pilot}"
MANIFEST="${MANIFEST:-${PROJECT_DIR}/manifests/part3_evolving_main_n10.csv}"
SOURCE_LOCK="${SOURCE_LOCK:-${PROJECT_DIR}/manifests/part3_evolving_main_execution_source.sha256}"
OUT_JSON="${OUT_JSON:-${PROJECT_DIR}/logs/abm_p3_evo10_pre_${SLURM_JOB_ID}.json}"

module purge
module load 2023
module load Python/3.11.3-GCCcore-12.3.0
source "${VENV_DIR}/bin/activate"

cd "${PROJECT_DIR}"
test -d "${EVIDENCE_DIR}"
test -f "${MANIFEST}"
test -f "${SOURCE_LOCK}"
sha256sum --check --strict "${SOURCE_LOCK}"

# These short broker runs exercise the deployed evolving-state path without a
# GPU. The manifest audit still checks every row of the 400-run design.
python3 scripts/validation/check_part3_closed_loop_preflight.py \
  --episode-log "${EVIDENCE_DIR}" \
  --manifest "${MANIFEST}" \
  --study-design main \
  --expected-seed-count 10 \
  --evolving-state \
  --evolving-state-interval-seconds 1800 \
  --duration 7200 \
  --warmup-seconds 3600 \
  --max-model-decisions 8 \
  --max-model-decisions-per-agent 3 \
  --decision-window-seconds 1800 \
  --max-model-decisions-per-window 4 \
  --guarded-path jobs/snellius_part3_qwen36_closed_loop_main_n10.sbatch \
  --guarded-path scripts/validation/verify_part3_evolving_pilot.py \
  --out-json "${OUT_JSON}"

echo "Passing evolving-persona main preflight: ${OUT_JSON}"
