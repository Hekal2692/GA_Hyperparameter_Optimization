#!/usr/bin/env bash
#SBATCH --job-name=budget_pair_sweep
#SBATCH --output=budget_pair_sweep_%j.out
#SBATCH --error=budget_pair_sweep_%j.err
#SBATCH --partition=long
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --time=5-00:00:00

set -euo pipefail

SUBMIT_DIR="${SLURM_SUBMIT_DIR:-$PWD}"
if [[ -f "${SUBMIT_DIR}/ga_config.json" ]]; then
  cd "$SUBMIT_DIR"
elif [[ -f "${SUBMIT_DIR}/Studienarbeit/ga_config.json" ]]; then
  cd "${SUBMIT_DIR}/Studienarbeit"
else
  echo "Could not find ga_config.json in ${SUBMIT_DIR} or ${SUBMIT_DIR}/Studienarbeit." >&2
  exit 1
fi

RUN_LOG_DIR="${HOME}/budget_pair_sweep_logs/${SLURM_JOB_ID:-manual}"
mkdir -p "$RUN_LOG_DIR"

RUN_CONFIG="${RUN_LOG_DIR}/ga_config.runtime.json"
python - "$RUN_LOG_DIR" "$RUN_CONFIG" <<'PY'
import json
import sys
from pathlib import Path

log_dir = Path(sys.argv[1]).resolve()
run_config = Path(sys.argv[2])

with open("ga_config.json", "r", encoding="utf-8") as handle:
    config = json.load(handle)

config.setdefault("paths", {})["log_dir"] = str(log_dir)

with open(run_config, "w", encoding="utf-8") as handle:
    json.dump(config, handle, indent=2)
PY

python - <<'PY'
import matplotlib
import networkx
import numpy
from deap import base, creator, tools
PY

python -u BudgetPairSweep.py \
  example_50T.json \
  example_70T.json \
  --config "$RUN_CONFIG" \
  --budgets 1800 2100 2400 2520 3360 \
  --cxpb-values 0.5 0.6 0.7 0.8 0.9 \
  --mutpb-values 0.1 0.15 0.2 0.25 0.3 \
  --shared-mutation-values 0.05 0.1 0.15 0.2 0.25 \
  --parallel-workers 2
