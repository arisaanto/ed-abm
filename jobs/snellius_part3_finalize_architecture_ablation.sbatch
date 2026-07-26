#!/bin/bash
# CPU-only finalization of already generated architecture-ablation responses.
# This corrects verifier scoping, verifies the immutable responses, and analyzes
# the 120 matched 2 x 2 sets. It does not load Qwen or rerun the ABM.

#SBATCH --job-name=abm_p3_abl_fin
#SBATCH --partition=rome
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=16G
#SBATCH --time=00:15:00
#SBATCH --output=logs/abm_p3_abl_fin_%j.out
#SBATCH --error=logs/abm_p3_abl_fin_%j.err

set -euo pipefail

PROJECT_DIR="${PROJECT_DIR:-${HOME}/ABM/ABM}"
RESULTS_DIR="${RESULTS_DIR:-${HOME}/ABM_results/part3_qwen36_architecture_ablation}"
PYTHON_MODULE="${PYTHON_MODULE:-Python/3.11.3-GCCcore-12.3.0}"

module purge
module load 2023
module load "${PYTHON_MODULE}"
cd "${PROJECT_DIR}"

PACKET_FILE="${RESULTS_DIR}/inputs/prompt_packets/part3_architecture_ablation_packets.jsonl"
RESPONSES_FILE="${RESULTS_DIR}/responses.jsonl"
if [[ ! -f "${PACKET_FILE}" || ! -f "${RESPONSES_FILE}" ]]; then
  echo "Missing preserved ablation packets or responses in ${RESULTS_DIR}" >&2
  exit 2
fi

python3 scripts/validation/verify_part3_vllm_responses.py \
  --input "${PACKET_FILE}" \
  --responses "${RESPONSES_FILE}" \
  --out-json "${RESULTS_DIR}/verification.json" \
  --reject-fixtures

python3 - "${RESULTS_DIR}/verification.json" <<'PY'
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
    raise SystemExit("Corrected architecture-ablation verification failed")
print("All 480 preserved ablation responses passed corrected technical verification")
PY

python3 scripts/analysis/analyze_part3_architecture_ablation.py \
  --packets "${PACKET_FILE}" \
  --responses "${RESPONSES_FILE}" \
  --output-dir "${RESULTS_DIR}/analysis"

python3 - "${RESULTS_DIR}/analysis/architecture_ablation_summary.json" <<'PY'
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
print("Architecture ablation finalized: 120 matched four-cell comparisons")
PY

sha256sum \
  "${PACKET_FILE}" \
  "${RESPONSES_FILE}" \
  "${RESULTS_DIR}/verification.json" \
  "${RESULTS_DIR}/analysis/architecture_ablation_pair_effects.csv" \
  > "${RESULTS_DIR}/result_sha256.txt"

python3 - <<'PY'
print("Finalization used no GPU and did not rerun the ABM.")
PY
