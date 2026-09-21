#!/bin/bash
# Warm-up audit: 2 workloads x 2 conditions x 4 cut-offs x 30 paired seeds.

#SBATCH --job-name=abm_p2_warmup30
#SBATCH --partition=rome
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=128
#SBATCH --mem=180G
#SBATCH --time=02:00:00
#SBATCH --output=logs/abm_p2_warmup30_%j.out
#SBATCH --error=logs/abm_p2_warmup30_%j.err

set -euo pipefail

PROJECT_DIR="${PROJECT_DIR:-${HOME}/ABM/ABM}"
VENV_DIR="${VENV_DIR:-${HOME}/vllm-env}"
MANIFEST="${MANIFEST:-${PROJECT_DIR}/manifests/part2_warmup_audit_n30.csv}"
SCRATCH_DIR="${SCRATCH_DIR:-/scratch-shared/${USER}/abm/part2_warmup_audit_n30_${SLURM_JOB_ID}}"
RESULTS_DIR="${RESULTS_DIR:-${HOME}/ABM_results/part2_warmup_audit_n30}"
MAX_WORKERS="${MAX_WORKERS:-112}"

module purge
module load 2023
module load Python/3.11.3-GCCcore-12.3.0
source "${VENV_DIR}/bin/activate"

export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1

mkdir -p "${SCRATCH_DIR}" "${RESULTS_DIR}" "${PROJECT_DIR}/logs"
cd "${PROJECT_DIR}"

test -f "${MANIFEST}"
python3 -c 'from src.metrics import compute_distance_components; from src.simulation import Simulation'

python3 - "${MANIFEST}" <<'PY'
from pathlib import Path
import csv
import sys

with Path(sys.argv[1]).open(newline="") as handle:
    rows = list(csv.DictReader(handle))
expected = {
    (scenario, condition, str(warmup), str(seed))
    for scenario in ("normal_load", "high_load_high_acuity")
    for condition in ("baseline", "cockpit_only")
    for warmup in (0, 3600, 7200, 10800)
    for seed in range(1, 31)
}
actual = {
    (row["scenario_mode"], row["condition"], row["warmup_seconds"], row["seed"])
    for row in rows
}
errors = []
if len(rows) != 480:
    errors.append(f"expected 480 rows, found {len(rows)}")
if actual != expected:
    errors.append("scenario-condition-warmup-seed grid is incomplete or duplicated")
if {row["duration"] for row in rows} != {"43200"}:
    errors.append("duration must be 43200 seconds")
if {row["scenario_start_hour"] for row in rows} != {"10"}:
    errors.append("scenario_start_hour must be 10")
if {row["isolate_exogenous_arrival_stream"].lower() for row in rows} != {"true"}:
    errors.append("arrival stream isolation must be enabled")
if errors:
    raise SystemExit("Manifest preflight failed: " + "; ".join(errors))
print("Manifest preflight passed: 480 paired warm-up-audit runs")
PY

python3 scripts/run/run_batch.py \
  --manifest "${MANIFEST}" \
  --max-workers "${MAX_WORKERS}" \
  --output-root "${SCRATCH_DIR}"

python3 - "${SCRATCH_DIR}" <<'PY'
from pathlib import Path
import json
import sys

root = Path(sys.argv[1])
summaries = list(root.rglob("summary.json"))
if len(summaries) != 480:
    raise SystemExit(f"Expected 480 summaries, found {len(summaries)}")
bad = []
for path in summaries:
    payload = json.loads(path.read_text())
    metadata = payload.get("metadata", {})
    if metadata.get("part3_exogenous_arrival_stream_isolated") is not True:
        bad.append(f"{path}: arrival stream not isolated")
    if int(metadata.get("warmup_seconds", -1)) not in {0, 3600, 7200, 10800}:
        bad.append(f"{path}: unexpected warm-up")
if bad:
    raise SystemExit(f"Warm-up audit verification failed: {bad[0]}")
print("Verified 480 warm-up-audit summaries")
PY

rsync -a "${SCRATCH_DIR}/" "${RESULTS_DIR}/"
echo "Results: ${RESULTS_DIR}"
