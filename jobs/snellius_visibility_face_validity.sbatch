#!/bin/bash
# Visibility face-validity and perception-envelope audit.
# Three perception envelopes x Baseline/transparent central station x 20 seeds.

#SBATCH --job-name=abm_visibility_audit
#SBATCH --partition=rome
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=128
#SBATCH --mem=180G
#SBATCH --time=01:00:00
#SBATCH --output=logs/abm_visibility_audit_%j.out
#SBATCH --error=logs/abm_visibility_audit_%j.err

set -euo pipefail

PROJECT_DIR="${PROJECT_DIR:-${HOME}/ABM/ABM}"
VENV_DIR="${VENV_DIR:-${HOME}/vllm-env}"
RESULTS_DIR="${RESULTS_DIR:-${HOME}/ABM_results/visibility_face_validity_n20}"

module purge
module load 2023
module load Python/3.11.3-GCCcore-12.3.0
source "${VENV_DIR}/bin/activate"

export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1

mkdir -p "${RESULTS_DIR}" "${PROJECT_DIR}/logs"
cd "${PROJECT_DIR}"

for required_file in \
  "${PROJECT_DIR}/config.py" \
  "${PROJECT_DIR}/data/wall_positions.json" \
  "${PROJECT_DIR}/data/zone_boundaries.json"; do
  test -f "${required_file}" || {
    echo "Missing required file: ${required_file}" >&2
    exit 2
  }
done

python3 -c 'from src.simulation import Simulation'

python3 scripts/analysis/audit_visibility_face_validity.py \
  --seeds 20 \
  --workers 112 \
  --output-dir "${RESULTS_DIR}"

test -s "${RESULTS_DIR}/visibility_face_validity_runs.csv"
test -s "${RESULTS_DIR}/visibility_face_validity_summary.md"
echo "Results: ${RESULTS_DIR}"
