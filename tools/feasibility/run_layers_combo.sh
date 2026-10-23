#!/usr/bin/env bash
# Layer-count test, mixed: only det_head or only map_head stopped after 4 layers.
# Inference latency (batch 1, repo, matmul) and full val eval (repo, matmul).
set -eo pipefail
cd "$(dirname "$0")/../.."
source .venv/bin/activate
export PYTHONPATH="$(pwd)${PYTHONPATH:+:$PYTHONPATH}"
CFG=tools/feasibility/configs
LOG=tools/feasibility/outputs/logs_layers
mkdir -p "$LOG"
for c in det4_map6 det6_map4; do
    python tools/feasibility/infer_bench.py --mode repo --queue matmul \
        --config $CFG/stage2_distribution_$c.py --tag matmul_$c > "$LOG/infer_$c.log" 2>&1
done
for c in det4_map6 det6_map4; do
    python tools/feasibility/queue_matmul_eval.py --variant matmul --run $c --no-save \
        --config $CFG/stage2_distribution_$c.py > "$LOG/eval_$c.log" 2>&1
done
echo ALL_DONE > "$LOG/done_combo"
