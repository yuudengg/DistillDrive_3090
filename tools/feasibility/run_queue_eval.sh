#!/usr/bin/env bash
# Exp A: original vs matmul InstanceQueue.prepare_motion, full val, repo mode.
# original_2 (same code, inference only) gives the run-to-run noise floor.
set -eo pipefail
cd "$(dirname "$0")/../.."
source .venv/bin/activate
export PYTHONPATH="$(pwd)${PYTHONPATH:+:$PYTHONPATH}"
Q=tools/feasibility/outputs/queue_eval
mkdir -p "$Q"
python tools/feasibility/queue_matmul_eval.py --variant original --run original > "$Q/original.log" 2>&1
python tools/feasibility/queue_matmul_eval.py --variant matmul --run matmul > "$Q/matmul.log" 2>&1
python tools/feasibility/queue_matmul_eval.py --variant original --run original_2 --no-metrics > "$Q/original_2.log" 2>&1
python tools/feasibility/compare_outputs.py "$Q/original/outputs.pkl" "$Q/original_2/outputs.pkl" --name noise_floor > "$Q/compare_noise_floor.log" 2>&1
python tools/feasibility/compare_outputs.py "$Q/original/outputs.pkl" "$Q/matmul/outputs.pkl" --name orig_vs_matmul > "$Q/compare_orig_vs_matmul.log" 2>&1
echo ALL_DONE > "$Q/done"
