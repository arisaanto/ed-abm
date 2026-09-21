#!/bin/bash
# Verify and analyse the preserved 600-response common-situation audit on CPU.

#SBATCH --job-name=abm_p3_common_post
#SBATCH --partition=rome
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=12G
#SBATCH --time=00:15:00
#SBATCH --output=logs/abm_p3_common_post_%j.out
#SBATCH --error=logs/abm_p3_common_post_%j.err

set -euo pipefail

PROJECT_DIR="${PROJECT_DIR:-${HOME}/ABM/ABM}"
INPUTS_DIR="${INPUTS_DIR:-${HOME}/ABM_results/part3_common_situation_audit_inputs}"
RESULTS_DIR="${RESULTS_DIR:-${HOME}/ABM_results/part3_common_situation_audit}"
PYTHON_MODULE="${PYTHON_MODULE:-Python/3.11.3-GCCcore-12.3.0}"

module purge
module load 2023
module load "${PYTHON_MODULE}"
cd "${PROJECT_DIR}"

PACKETS="${INPUTS_DIR}/prompt_packets/part3_common_situation_audit_packets.jsonl"
RESPONSES="${RESULTS_DIR}/responses.jsonl"
test -f "${PACKETS}"
test -f "${RESPONSES}"

python3 scripts/validation/verify_part3_vllm_responses.py \
  --input "${PACKETS}" \
  --responses "${RESPONSES}" \
  --reject-fixtures \
  --scientific-episode-count 120 \
  --out-json "${RESULTS_DIR}/verification.json"

python3 scripts/analysis/analyze_part3_common_situation_audit.py \
  --packets "${PACKETS}" \
  --responses "${RESPONSES}" \
  --output-dir "${RESULTS_DIR}/analysis"

python3 - "${RESULTS_DIR}/verification.json" "${RESULTS_DIR}/analysis/common_situation_audit_summary.json" <<'PY'
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
    raise SystemExit("Common-situation audit post-processing failed")
print("Common-situation audit is technically complete")
print("Prespecified directions all passed:", analysis.get("all_prespecified_directions_pass"))
PY

sha256sum \
  "${PACKETS}" \
  "${RESPONSES}" \
  "${RESULTS_DIR}/verification.json" \
  "${RESULTS_DIR}/analysis/common_situation_audit_summary.json" \
  > "${RESULTS_DIR}/result_sha256.txt"
