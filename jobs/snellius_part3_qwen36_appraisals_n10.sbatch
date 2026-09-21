#!/bin/bash
# Post-main synthetic appraisal endpoint. This does not rerun or modify the ABM.
# It consumes 800 CPU-preflighted paired packets and generates one survey bundle
# plus one interview bundle per selected shift with one Qwen engine load.
# Maximum estimate: 4 A100 x 2.0 h x 128 = 1,024 GPU SBU. Actual charging follows
# observed runtime; the extra ceiling avoids losing a valid run to wall time.

#SBATCH --job-name=abm_p3_appraise
#SBATCH --partition=gpu_a100
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --mem=460G
#SBATCH --gpus=4
#SBATCH --time=02:00:00
#SBATCH --output=logs/abm_p3_appraise_%j.out
#SBATCH --error=logs/abm_p3_appraise_%j.err

set -euo pipefail

PROJECT_DIR="${PROJECT_DIR:-${HOME}/ABM/ABM}"
VENV_DIR="${VENV_DIR:-${HOME}/vllm-env}"
MAIN_RESULTS_DIR="${MAIN_RESULTS_DIR:-${HOME}/ABM_results/part3_evolving_main_n10}"
APPRAISAL_INPUTS_DIR="${APPRAISAL_INPUTS_DIR:-${HOME}/ABM_results/part3_evolving_appraisal_inputs_n10}"
SCRATCH_DIR="${SCRATCH_DIR:-/scratch-shared/${USER}/abm/part3_evolving_qwen36_appraisals_n10_${SLURM_JOB_ID}}"
RESULTS_DIR="${RESULTS_DIR:-${HOME}/ABM_results/part3_evolving_qwen36_appraisals_n10}"
HF_HOME="${HF_HOME:-/scratch-shared/${USER}/huggingface}"
PYTHON_MODULE="${PYTHON_MODULE:-Python/3.11.3-GCCcore-12.3.0}"
CUDA_MODULE="${CUDA_MODULE:-}"
MODEL_ID="${MODEL_ID:-Qwen/Qwen3.6-35B-A3B}"
MODEL_CACHE_DIR="${HF_HOME}/hub/models--Qwen--Qwen3.6-35B-A3B"
MODEL_REVISION="${MODEL_REVISION:-995ad96eacd98c81ed38be0c5b274b04031597b0}"
RESUME_FROM="${RESUME_FROM:-}"
SEMANTIC_RETRIES="${SEMANTIC_RETRIES:-}"
if [[ -z "${SEMANTIC_RETRIES}" ]]; then
  if [[ -n "${RESUME_FROM}" ]]; then
    SEMANTIC_RETRIES=3
  else
    SEMANTIC_RETRIES=2
  fi
fi
if [[ ! "${SEMANTIC_RETRIES}" =~ ^[0-9]+$ ]]; then
  echo "SEMANTIC_RETRIES must be a non-negative integer" >&2
  exit 2
fi

module purge
module load 2023
module load "${PYTHON_MODULE}"
if [[ -n "${CUDA_MODULE}" ]]; then module load "${CUDA_MODULE}"; fi
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

mkdir -p "${HF_HOME}" "${SCRATCH_DIR}" "${RESULTS_DIR}"
cd "${PROJECT_DIR}"

sync_results() {
  rsync -a "${SCRATCH_DIR}/" "${RESULTS_DIR}/"
}
trap sync_results EXIT

if [[ ! -f "${MAIN_RESULTS_DIR}/evolving_main_integrity.json" ]]; then
  echo "Missing completed main-study integrity file: ${MAIN_RESULTS_DIR}" >&2
  exit 2
fi
if [[ ! -f "${APPRAISAL_INPUTS_DIR}/appraisal_packet_preflight.json" ]]; then
  echo "Missing CPU-verified appraisal packets: ${APPRAISAL_INPUTS_DIR}" >&2
  exit 2
fi
if [[ -n "${RESUME_FROM}" && ! -f "${RESUME_FROM}" ]]; then
  echo "Missing previous response JSONL for repair: ${RESUME_FROM}" >&2
  exit 2
fi
if find "${SCRATCH_DIR}" -mindepth 1 -maxdepth 1 -print -quit | grep -q .; then
  echo "Scratch output is not empty: ${SCRATCH_DIR}" >&2
  exit 2
fi
if find "${RESULTS_DIR}" -mindepth 1 -maxdepth 1 -print -quit | grep -q .; then
  echo "Results output is not empty: ${RESULTS_DIR}" >&2
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
    "agent_experience_export_pass",
    "experience_event_export_pass",
    "spatial_experience_map_export_pass",
    "scientific_result",
)
failed = [key for key in required if result.get(key) is not True]
if (
    failed
    or result.get("observed_summary_count") != 400
    or outer.get("verification_pass") is not True
):
    raise SystemExit(
        "Main study is not eligible for appraisal generation: " + ", ".join(failed)
    )
print("Accepted complete 400-run main study for post-run appraisals")
PY
cp "${MAIN_RESULTS_DIR}/evolving_main_integrity.json" "${SCRATCH_DIR}/accepted_main_integrity.json"
if [[ -z "${RESUME_FROM}" ]]; then
  (
    cd "${PROJECT_DIR}"
    sha256sum -c "${APPRAISAL_INPUTS_DIR}/source_sha256.txt"
  )
else
  echo "Repair mode: preserving the immutable v3 packets and revalidating prior raw responses"
fi
cp -a "${APPRAISAL_INPUTS_DIR}" "${SCRATCH_DIR}/inputs"

python3 - "${SCRATCH_DIR}/inputs/appraisal_packet_preflight.json" <<'PY'
import json
from pathlib import Path
import sys

result = json.loads(Path(sys.argv[1]).read_text())
selection = result.get("selection") or {}
experience_audit = result.get("experience_evidence_audit") or {}
if (
    result.get("preflight_pass") is not True
    or result.get("packet_count") != 800
    or result.get("packet_type_counts")
    != {"end_of_shift_interview_bundle": 400, "end_of_shift_survey_bundle": 400}
    or selection.get("paired_condition_sampling") is not True
    or selection.get("selected_paired_unit_count") != 100
    or selection.get("selected_seed_count_per_scenario_persona") != 10
    or selection.get("expected_seed_stratum_count") != 100
    or selection.get("observed_seed_stratum_count") != 100
    or selection.get("conditions_per_paired_unit") != 4
    or selection.get("incomplete_pair_count") != 0
    or experience_audit.get("audit_pass") is not True
    or experience_audit.get("missing_phase_shift_count") != 0
    or experience_audit.get("duplicate_place_label_shift_count") != 0
    or experience_audit.get("self_transition_shift_count") != 0
    or experience_audit.get("redundant_travel_wording_shift_count") != 0
    or experience_audit.get("uninformative_travel_shift_count") != 0
    or experience_audit.get("internal_label_shift_count") != 0
    or experience_audit.get("incomplete_accumulated_experience_shift_count") != 0
    or result.get("maximum_estimated_tokens", 99999) + 4096 > 12288
):
    raise SystemExit("Balanced appraisal packet preflight did not pass")
print("Balanced paired appraisal packet preflight passed")
PY

PACKET_FILE="${SCRATCH_DIR}/inputs/prompt_packets/end_of_shift_appraisal_prompt_packets.jsonl"
RESUME_ARGS=()
if [[ -n "${RESUME_FROM}" ]]; then
  RESUME_ARGS=(--resume-from "${RESUME_FROM}")
fi
python3 scripts/run/run_part3_vllm_offline.py \
  --input "${PACKET_FILE}" \
  --output "${SCRATCH_DIR}/responses.jsonl" \
  --model "${MODEL_ID}" \
  --model-revision "${MODEL_REVISION:-}" \
  --tensor-parallel-size 4 \
  --max-model-len 12288 \
  --max-output-tokens 4096 \
  --batch-size 16 \
  --semantic-retries "${SEMANTIC_RETRIES}" \
  --reject-fixtures \
  "${RESUME_ARGS[@]}" \
  --summary-json "${SCRATCH_DIR}/inference_preflight.json" \
  --dry-run

if [[ -n "${RESUME_FROM}" ]]; then
  python3 - "${SCRATCH_DIR}/inference_preflight.json" <<'PY'
import json
from pathlib import Path
import sys

audit = json.loads(Path(sys.argv[1]).read_text()).get("resume_audit") or {}
reusable = int(audit.get("locally_reusable_response_count", 0))
repair = int(audit.get("repair_packet_count", 0))
if reusable + repair != 800 or reusable <= 0 or repair <= 0 or repair >= 800:
    raise SystemExit(
        f"Repair preflight refused unsafe inventory: reusable={reusable}, repair={repair}"
    )
print(f"Repair preflight passed: preserving {reusable}; regenerating only {repair}")
PY
fi

if [[ -z "${MODEL_REVISION}" && -f "${MODEL_CACHE_DIR}/refs/main" ]]; then
  MODEL_REVISION="$(tr -d '[:space:]' < "${MODEL_CACHE_DIR}/refs/main")"
fi
if [[ -z "${MODEL_REVISION}" ]]; then
  echo "Set MODEL_REVISION or restore the cached model revision." >&2
  exit 2
fi

python3 - <<'PY'
import torch
if not torch.cuda.is_available() or torch.cuda.device_count() != 4:
    raise SystemExit("GPU preflight failed: expected four A100 GPUs")
PY

sha256sum \
  src/interviews.py \
  src/personas.py \
  src/vllm_backend.py \
  manifests/part3_appraisal_analysis_spec.json \
  scripts/analysis/analyze_part3_appraisals.py \
  scripts/build/build_part3_appraisal_packets.py \
  scripts/build/plan_part3_synthetic_study.py \
  scripts/run/run_part3_vllm_offline.py \
  scripts/validation/verify_part3_vllm_responses.py \
  jobs/snellius_part3_qwen36_appraisals_n10.sbatch \
  > "${SCRATCH_DIR}/source_sha256.txt"

# Thinking is intentionally disabled. The endpoint requires concise,
# trace-grounded interpretation rather than chain-of-thought generation.
python3 scripts/run/run_part3_vllm_offline.py \
  --input "${PACKET_FILE}" \
  --output "${SCRATCH_DIR}/responses.jsonl" \
  --model "${MODEL_ID}" \
  --model-revision "${MODEL_REVISION}" \
  --tensor-parallel-size 4 \
  --max-model-len 12288 \
  --max-output-tokens 4096 \
  --batch-size 16 \
  --semantic-retries "${SEMANTIC_RETRIES}" \
  --reject-fixtures \
  "${RESUME_ARGS[@]}"

python3 scripts/validation/verify_part3_vllm_responses.py \
  --input "${PACKET_FILE}" \
  --responses "${SCRATCH_DIR}/responses.jsonl" \
  --out-json "${SCRATCH_DIR}/verification.json" \
  --reject-fixtures

python3 - "${SCRATCH_DIR}/verification.json" <<'PY'
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
    raise SystemExit("Appraisal technical verification failed")
print(
    "Appraisals passed technical checks; human review remains mandatory; "
    f"semantic retries={result.get('semantic_retry_total', 0)} across "
    f"{result.get('semantic_retry_packet_count', 0)} packets"
)
PY

python3 scripts/analysis/analyze_part3_appraisals.py \
  --packets "${PACKET_FILE}" \
  --responses "${SCRATCH_DIR}/responses.jsonl" \
  --verification "${SCRATCH_DIR}/verification.json" \
  --output-dir "${SCRATCH_DIR}/analysis"
