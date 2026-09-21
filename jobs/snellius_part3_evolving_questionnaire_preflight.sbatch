#!/bin/bash
# CPU-only construction and audit of the questionnaire construct translation.
# No ABM rerun, GPU use, Qwen load, or hospital questionnaire data.

#SBATCH --job-name=abm_p3_q_pre
#SBATCH --partition=rome
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=16G
#SBATCH --time=00:30:00
#SBATCH --output=logs/abm_p3_q_pre_%j.out
#SBATCH --error=logs/abm_p3_q_pre_%j.err

set -euo pipefail

PROJECT_DIR="${PROJECT_DIR:-${HOME}/ABM/ABM}"
MAIN_RESULTS_DIR="${MAIN_RESULTS_DIR:-${HOME}/ABM_results/part3_evolving_main_n10}"
RESULTS_DIR="${RESULTS_DIR:-${HOME}/ABM_results/part3_evolving_questionnaire_audit_inputs}"
PYTHON_MODULE="${PYTHON_MODULE:-Python/3.11.3-GCCcore-12.3.0}"
MODEL_ID="${MODEL_ID:-Qwen/Qwen3.6-35B-A3B}"

module purge
module load 2023
module load "${PYTHON_MODULE}"

cd "${PROJECT_DIR}"
if [[ -e "${RESULTS_DIR}" ]]; then
  echo "Questionnaire inputs already exist: ${RESULTS_DIR}" >&2
  exit 2
fi
if [[ ! -f "${MAIN_RESULTS_DIR}/evolving_main_integrity.json" ]]; then
  echo "Missing completed main-study integrity file" >&2
  exit 2
fi

python3 - "${MAIN_RESULTS_DIR}/evolving_main_integrity.json" <<'PY'
import json
from pathlib import Path
import sys

outer = json.loads(Path(sys.argv[1]).read_text())
result = outer.get("base_verification") or outer
required = (
    "verification_pass",
    "technical_integrity_pass",
    "main_design_complete",
    "study_design_complete",
    "persona_exposure_balance_pass",
    "scientific_result",
)
failed = [key for key in required if result.get(key) is not True]
if (
    failed
    or result.get("observed_summary_count") != 400
    or outer.get("verification_pass") is not True
):
    raise SystemExit("Main study is not eligible for the questionnaire audit: " + ", ".join(failed))
print("Accepted complete 400-run main study; the ABM will not be rerun")
PY

python3 scripts/build/build_part3_questionnaire_audit_packets.py \
  --results-root "${MAIN_RESULTS_DIR}/main" \
  --output-dir "${RESULTS_DIR}"

PACKET_FILE="${RESULTS_DIR}/prompt_packets/questionnaire_audit_prompt_packets.jsonl"
python3 scripts/run/run_part3_vllm_offline.py \
  --input "${PACKET_FILE}" \
  --output "${RESULTS_DIR}/responses.jsonl" \
  --model "${MODEL_ID}" \
  --tensor-parallel-size 4 \
  --max-model-len 8192 \
  --max-output-tokens 2048 \
  --batch-size 16 \
  --semantic-retries 2 \
  --reject-fixtures \
  --summary-json "${RESULTS_DIR}/offline_dry_run.json" \
  --dry-run

sha256sum \
  src/cognitive_policy.py \
  src/interviews.py \
  src/personas.py \
  src/questionnaire_audit.py \
  src/vllm_backend.py \
  scripts/analysis/analyze_part3_questionnaire_audit.py \
  scripts/build/build_part3_appraisal_packets.py \
  scripts/build/build_part3_questionnaire_audit_packets.py \
  scripts/run/run_part3_vllm_offline.py \
  scripts/validation/verify_part3_questionnaire_audit.py \
  jobs/snellius_part3_qwen36_evolving_questionnaire_audit.sbatch \
  > "${RESULTS_DIR}/source_sha256.txt"

python3 - \
  "${RESULTS_DIR}/questionnaire_audit_preflight.json" \
  "${RESULTS_DIR}/offline_dry_run.json" <<'PY'
import json
from pathlib import Path
import sys

design = json.loads(Path(sys.argv[1]).read_text())
dry_run = json.loads(Path(sys.argv[2]).read_text())
if (
    design.get("preflight_pass") is not True
    or design.get("questionnaire_audit_version") != "construct_translation_observed_outcomes"
    or design.get("packet_count") != 390
    or design.get("control_variant_counts") != {
        "evidence_withheld_neutral": 30,
        "evidence_withheld_persona": 30,
        "full_trace_neutral": 30,
        "full_trace_persona": 300,
    }
    or design.get("assessment_variant_counts") != {
        "balanced": 390,
    }
    or design.get("matched_control_stratum_count") != 30
    or design.get("simulation_window_duration_seconds") != [36000]
    or design.get("q2_construct_sanity_pass") is not True
    or design.get("measurement_discriminability", {}).get("audit_pass") is not True
    or design.get("confidential_hospital_data_in_packets") is not False
    or design.get("questionnaire_answers_loaded") is not False
    or dry_run.get("packet_count") != 390
    or dry_run.get("fixtures_rejected") is not True
    or dry_run.get("maximum_estimated_input_tokens", 99999) + 2048 > 8192
):
    raise SystemExit("Questionnaire CPU preflight failed")
print("Questionnaire CPU preflight passed: 390 blinded bundles")
PY
