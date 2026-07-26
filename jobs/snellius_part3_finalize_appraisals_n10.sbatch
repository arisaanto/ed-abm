#!/bin/bash
# CPU-only finalization, verification, and analysis of the Part 3 appraisals.
# No ABM run, model load, or LLM inference occurs in this job.

#SBATCH --job-name=abm_p3_app_final
#SBATCH --partition=rome
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=16G
#SBATCH --time=00:20:00
#SBATCH --output=logs/abm_p3_app_final_%j.out
#SBATCH --error=logs/abm_p3_app_final_%j.err

set -euo pipefail

PROJECT_DIR="${PROJECT_DIR:-${HOME}/ABM/ABM}"
INPUT_RESPONSES="${INPUT_RESPONSES:?Set INPUT_RESPONSES to the v3 repair responses.jsonl}"
PACKET_FILE="${PACKET_FILE:?Set PACKET_FILE to the immutable appraisal packet JSONL}"
RESULTS_DIR="${RESULTS_DIR:-${HOME}/ABM_results/part3_qwen36_appraisals_n10_final}"
PYTHON_MODULE="${PYTHON_MODULE:-Python/3.11.3-GCCcore-12.3.0}"

module purge
module load 2023
module load "${PYTHON_MODULE}"

cd "${PROJECT_DIR}"
if [[ ! -f "${INPUT_RESPONSES}" || ! -f "${PACKET_FILE}" ]]; then
  echo "Missing repair responses or immutable packets" >&2
  exit 2
fi
if [[ -e "${RESULTS_DIR}" ]]; then
  echo "Final appraisal output already exists: ${RESULTS_DIR}" >&2
  exit 2
fi
mkdir -p "${RESULTS_DIR}"

python3 scripts/build/finalize_part3_appraisal_responses.py \
  --input "${INPUT_RESPONSES}" \
  --packets "${PACKET_FILE}" \
  --output "${RESULTS_DIR}/responses.jsonl" \
  --audit-json "${RESULTS_DIR}/deterministic_postprocessing_audit.json"

python3 scripts/validation/verify_part3_vllm_responses.py \
  --input "${PACKET_FILE}" \
  --responses "${RESULTS_DIR}/responses.jsonl" \
  --out-json "${RESULTS_DIR}/verification.json" \
  --reject-fixtures

python3 - "${RESULTS_DIR}/verification.json" <<'PY'
import json
from pathlib import Path
import sys

result = json.loads(Path(sys.argv[1]).read_text())
if (
    result.get("technical_verification_pass") is not True
    or result.get("schema_or_grounding_error_count") != 0
    or result.get("missing_response_count") != 0
    or result.get("unexpected_response_count") != 0
    or result.get("semantic_correction_exhausted_count") != 0
    or result.get("appraisal_packet_counts")
    != {"end_of_shift_interview_bundle": 400, "end_of_shift_survey_bundle": 400}
    or result.get("appraisal_human_review_required") is not True
):
    raise SystemExit("Final appraisal technical verification failed")
print("Final appraisals passed technical verification")
PY

python3 scripts/analysis/analyze_part3_appraisals.py \
  --packets "${PACKET_FILE}" \
  --responses "${RESULTS_DIR}/responses.jsonl" \
  --verification "${RESULTS_DIR}/verification.json" \
  --output-dir "${RESULTS_DIR}/analysis"

python3 - "${RESULTS_DIR}/analysis/appraisal_analysis_summary.json" <<'PY'
import json
from pathlib import Path
import sys

result = json.loads(Path(sys.argv[1]).read_text())
expected_estimable = {
    "team_awareness",
    "interruption_burden",
    "task_continuity",
    "patient_accessibility",
    "spatial_legibility",
    "overall_person_space_fit",
}
if (
    result.get("analysis_pass") is not True
    or result.get("survey_bundle_count") != 400
    or result.get("interview_bundle_count") != 400
    or result.get("survey_response_row_count") != 2800
    or result.get("interview_answer_row_count") != 2400
    or result.get("paired_analysis_ready_for_fully_estimable_dimensions") is not True
    or set(result.get("fully_estimable_dimensions", [])) != expected_estimable
    or result.get("partially_estimable_dimensions") != ["privacy_and_control"]
    or result.get("non_estimable_dimensions") != []
    or result.get("manual_review_required") is not True
):
    raise SystemExit("Final appraisal analysis audit failed")
print(
    "Six dimensions are fully paired; privacy/control is transparently "
    "only partially estimable from the simulated evidence"
)
PY

sha256sum \
  scripts/analysis/analyze_part3_appraisals.py \
  scripts/build/finalize_part3_appraisal_responses.py \
  scripts/validation/verify_part3_vllm_responses.py \
  src/interviews.py \
  src/vllm_backend.py \
  > "${RESULTS_DIR}/source_sha256.txt"
