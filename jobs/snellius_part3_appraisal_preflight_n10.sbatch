#!/bin/bash
# CPU-only construction and audit of the paired Part 3 appraisal packets.
# No ABM run, model load, or LLM inference occurs in this job.

#SBATCH --job-name=abm_p3_app_pre
#SBATCH --partition=rome
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=16G
#SBATCH --time=00:20:00
#SBATCH --output=logs/abm_p3_app_pre_%j.out
#SBATCH --error=logs/abm_p3_app_pre_%j.err

set -euo pipefail

PROJECT_DIR="${PROJECT_DIR:-${HOME}/ABM/ABM}"
MAIN_RESULTS_DIR="${MAIN_RESULTS_DIR:-${HOME}/ABM_results/part3_closed_loop_main_n10}"
RESULTS_DIR="${RESULTS_DIR:-${HOME}/ABM_results/part3_appraisal_inputs_n10}"
PYTHON_MODULE="${PYTHON_MODULE:-Python/3.11.3-GCCcore-12.3.0}"
REPAIR_RESPONSES="${REPAIR_RESPONSES:-}"
EXISTING_INPUTS_DIR="${EXISTING_INPUTS_DIR:-}"

module purge
module load 2023
module load "${PYTHON_MODULE}"

mkdir -p "$(dirname "${RESULTS_DIR}")"
cd "${PROJECT_DIR}"

if [[ -n "${REPAIR_RESPONSES}" ]]; then
  if [[ -z "${EXISTING_INPUTS_DIR}" ]]; then
    echo "Repair preflight requires EXISTING_INPUTS_DIR" >&2
    exit 2
  fi
  PACKET_FILE="${EXISTING_INPUTS_DIR}/prompt_packets/end_of_shift_appraisal_prompt_packets.jsonl"
  if [[ ! -f "${PACKET_FILE}" || ! -f "${REPAIR_RESPONSES}" ]]; then
    echo "Repair preflight is missing packets or prior responses" >&2
    exit 2
  fi
  if [[ -e "${RESULTS_DIR}" ]]; then
    echo "Repair preflight output already exists: ${RESULTS_DIR}" >&2
    exit 2
  fi
  mkdir -p "${RESULTS_DIR}"
  python3 scripts/run/run_part3_vllm_offline.py \
    --input "${PACKET_FILE}" \
    --output "${RESULTS_DIR}/responses.jsonl" \
    --resume-from "${REPAIR_RESPONSES}" \
    --reject-fixtures \
    --summary-json "${RESULTS_DIR}/repair_preflight.json" \
    --dry-run
  python3 - "${RESULTS_DIR}/repair_preflight.json" <<'PY'
import json
from pathlib import Path
import sys

result = json.loads(Path(sys.argv[1]).read_text())
audit = result.get("resume_audit") or {}
reusable = int(audit.get("locally_reusable_response_count", 0))
repair = int(audit.get("repair_packet_count", 0))
if result.get("packet_count") != 800 or reusable + repair != 800:
    raise SystemExit(
        f"Repair inventory is incomplete: reusable={reusable}, repair={repair}"
    )
if reusable <= 0 or repair <= 0 or repair >= 800:
    raise SystemExit(
        f"Repair inventory is unsafe: reusable={reusable}, repair={repair}"
    )
print(f"CPU repair preflight passed: preserve {reusable}; regenerate {repair}")
PY
  exit 0
fi

if [[ ! -f "${MAIN_RESULTS_DIR}/integrity_check.json" ]]; then
  echo "Missing completed main-study integrity file: ${MAIN_RESULTS_DIR}" >&2
  exit 2
fi
if [[ -e "${RESULTS_DIR}" ]]; then
  echo "Appraisal input output already exists: ${RESULTS_DIR}" >&2
  exit 2
fi

python3 - "${MAIN_RESULTS_DIR}/integrity_check.json" <<'PY'
import json
from pathlib import Path
import sys

result = json.loads(Path(sys.argv[1]).read_text())
required = (
    "verification_pass",
    "technical_integrity_pass",
    "main_design_complete",
    "study_design_complete",
    "persona_exposure_balance_pass",
    "agent_experience_export_pass",
    "experience_event_export_pass",
    "spatial_experience_map_export_pass",
    "scientific_result",
)
failed = [key for key in required if result.get(key) is not True]
if failed or result.get("observed_summary_count") != 400:
    raise SystemExit(
        "Main study is not eligible for appraisal construction: "
        + ", ".join(failed)
    )
print("Accepted complete 400-run main study for appraisal packet construction")
PY

python3 scripts/build/build_part3_appraisal_packets.py \
  --results-root "${MAIN_RESULTS_DIR}/main" \
  --output-dir "${RESULTS_DIR}" \
  --max-events 18

sha256sum \
  src/cognitive_policy.py \
  src/interviews.py \
  src/personas.py \
  src/vllm_backend.py \
  scripts/build/build_part3_appraisal_packets.py \
  > "${RESULTS_DIR}/source_sha256.txt"

python3 - "${RESULTS_DIR}/appraisal_packet_preflight.json" <<'PY'
import json
from pathlib import Path
import sys

result = json.loads(Path(sys.argv[1]).read_text())
selection = result.get("selection") or {}
audit = result.get("experience_evidence_audit") or {}
if (
    result.get("preflight_pass") is not True
    or result.get("packet_count") != 800
    or result.get("packet_type_counts")
    != {"end_of_shift_interview_bundle": 400, "end_of_shift_survey_bundle": 400}
    or selection.get("selected_paired_unit_count") != 100
    or selection.get("selected_seed_count_per_scenario_persona") != 10
    or selection.get("expected_seed_stratum_count") != 100
    or selection.get("observed_seed_stratum_count") != 100
    or selection.get("conditions_per_paired_unit") != 4
    or selection.get("incomplete_pair_count") != 0
    or audit.get("audit_pass") is not True
    or audit.get("missing_phase_shift_count") != 0
    or audit.get("duplicate_place_label_shift_count") != 0
    or audit.get("self_transition_shift_count") != 0
    or audit.get("redundant_travel_wording_shift_count") != 0
    or audit.get("uninformative_travel_shift_count") != 0
    or audit.get("internal_label_shift_count") != 0
):
    raise SystemExit("Appraisal packet preflight failed")
print("CPU appraisal packet preflight passed")
PY
