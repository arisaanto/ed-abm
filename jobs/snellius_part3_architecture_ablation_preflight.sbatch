#!/bin/bash
# CPU-only construction and audit of the matched persona x memory ablation.
# This job does not rerun the ABM, load Qwen, or use a GPU.

#SBATCH --job-name=abm_p3_abl_pre
#SBATCH --partition=rome
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=16G
#SBATCH --time=00:20:00
#SBATCH --output=logs/abm_p3_abl_pre_%j.out
#SBATCH --error=logs/abm_p3_abl_pre_%j.err

set -euo pipefail

PROJECT_DIR="${PROJECT_DIR:-${HOME}/ABM/ABM}"
MAIN_RESULTS_DIR="${MAIN_RESULTS_DIR:-${HOME}/ABM_results/part3_closed_loop_main_n10}"
RESULTS_DIR="${RESULTS_DIR:-${HOME}/ABM_results/part3_architecture_ablation_inputs}"
PYTHON_MODULE="${PYTHON_MODULE:-Python/3.11.3-GCCcore-12.3.0}"
MODEL_ID="${MODEL_ID:-Qwen/Qwen3.6-35B-A3B}"

module purge
module load 2023
module load "${PYTHON_MODULE}"

cd "${PROJECT_DIR}"
if [[ -e "${RESULTS_DIR}" ]]; then
  echo "Ablation input output already exists: ${RESULTS_DIR}" >&2
  exit 2
fi
if [[ ! -f "${MAIN_RESULTS_DIR}/integrity_check.json" ]]; then
  echo "Missing completed main-study integrity file" >&2
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
    "scientific_result",
)
failed = [key for key in required if result.get(key) is not True]
if failed or result.get("observed_summary_count") != 400:
    raise SystemExit("Main study is not eligible for ablation: " + ", ".join(failed))
print("Accepted complete 400-run main study for matched architecture ablation")
PY

python3 scripts/build/build_part3_ablation_packets.py \
  --results-root "${MAIN_RESULTS_DIR}/main" \
  --output-dir "${RESULTS_DIR}" \
  --replicates-per-stratum 1

PACKET_FILE="${RESULTS_DIR}/prompt_packets/part3_architecture_ablation_packets.jsonl"
python3 scripts/run/run_part3_vllm_offline.py \
  --input "${PACKET_FILE}" \
  --output "${RESULTS_DIR}/responses.jsonl" \
  --model "${MODEL_ID}" \
  --tensor-parallel-size 4 \
  --max-model-len 8192 \
  --max-output-tokens 512 \
  --batch-size 32 \
  --semantic-retries 2 \
  --reject-fixtures \
  --summary-json "${RESULTS_DIR}/inference_preflight.json" \
  --dry-run

sha256sum \
  src/personas.py \
  src/vllm_backend.py \
  scripts/analysis/analyze_part3_architecture_ablation.py \
  scripts/build/build_part3_ablation_packets.py \
  scripts/build/plan_part3_synthetic_study.py \
  scripts/run/run_part3_vllm_offline.py \
  scripts/validation/verify_part3_vllm_responses.py \
  jobs/snellius_part3_qwen36_architecture_ablation.sbatch \
  > "${RESULTS_DIR}/source_sha256.txt"

python3 - \
  "${RESULTS_DIR}/part3_architecture_ablation_preflight.json" \
  "${RESULTS_DIR}/inference_preflight.json" <<'PY'
import json
from pathlib import Path
import sys

design = json.loads(Path(sys.argv[1]).read_text())
inference = json.loads(Path(sys.argv[2]).read_text())
if (
    design.get("preflight_pass") is not True
    or design.get("selected_source_count") != 120
    or design.get("packet_count") != 480
    or design.get("variant_counts") != {
        "full_persona_full_memory": 120,
        "neutral_no_memory": 120,
        "neutral_orientation": 120,
        "persona_no_memory": 120,
    }
    or design.get("source_model_counts") != {"Qwen/Qwen3.6-35B-A3B": 120}
    or len(design.get("source_model_revision_counts") or {}) != 1
    or design.get("source_thinking_enabled_counts") != {"false": 120}
    or design.get("source_seed_counts")
    != {str(seed): 12 for seed in range(1, 11)}
    or design.get("expected_source_count_per_seed") != 12
    or set((design.get("source_reference_action_counts") or {}).keys())
    != {"engage", "defer", "decline"}
    or design.get("source_memory_minimum", 0) <= 0
    or inference.get("packet_count") != 480
    or inference.get("thinking_enabled") is not False
    or inference.get("fixtures_rejected") is not True
    or inference.get("maximum_estimated_input_tokens", 99999) + 512 > 8192
):
    raise SystemExit("Architecture-ablation preflight failed")
print("CPU architecture-ablation preflight passed: 120 matched 2 x 2 sets")
PY
