#!/bin/bash
# Build and verify the frozen common-situation audit packets on CPU.

#SBATCH --job-name=abm_p3_common_pre
#SBATCH --partition=rome
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=00:20:00
#SBATCH --output=logs/abm_p3_common_pre_%j.out
#SBATCH --error=logs/abm_p3_common_pre_%j.err

set -euo pipefail

PROJECT_DIR="${PROJECT_DIR:-${HOME}/ABM/ABM}"
SOURCE_RESULTS="${SOURCE_RESULTS:-${HOME}/ABM_results/part3_evolving_main_n10/main}"
OUTPUT_DIR="${OUTPUT_DIR:-${HOME}/ABM_results/part3_common_situation_audit_inputs}"
PYTHON_MODULE="${PYTHON_MODULE:-Python/3.11.3-GCCcore-12.3.0}"

module purge
module load 2023
module load "${PYTHON_MODULE}"
mkdir -p "${PROJECT_DIR}/logs"
if [[ -e "${OUTPUT_DIR}" ]]; then
  echo "Refusing to overwrite existing output: ${OUTPUT_DIR}" >&2
  exit 2
fi
cd "${PROJECT_DIR}"
python3 scripts/build/build_part3_common_situation_audit.py \
  --results-root "${SOURCE_RESULTS}" \
  --output-dir "${OUTPUT_DIR}"
sha256sum \
  scripts/build/build_part3_common_situation_audit.py \
  scripts/analysis/analyze_part3_common_situation_audit.py \
  "${OUTPUT_DIR}/prompt_packets/part3_common_situation_audit_packets.jsonl" \
  "${OUTPUT_DIR}/part3_common_situation_audit_inventory.csv" \
  "${OUTPUT_DIR}/part3_common_situation_audit_spec.json" \
  > "${OUTPUT_DIR}/source_sha256.txt"
