#!/usr/bin/env bash
# Layer-count test (no retraining): det/map heads stopped after 4 or 5 decoder layers.
# Training memory (batch 4, 60 iters, with_cp on, matmul queue), inference latency
# (batch 1, repo, matmul; 6 layers re-measured as baseline), full val eval (repo, matmul).
set -eo pipefail
cd "$(dirname "$0")/../.."
source .venv/bin/activate
export PYTHONPATH="$(pwd)${PYTHONPATH:+:$PYTHONPATH}"
export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:128
CFG=tools/feasibility/configs
LOG=tools/feasibility/outputs/logs_layers
mkdir -p "$LOG"
for n in 4 5; do
    python tools/feasibility/train_memory.py --iters 60 --with-cp on --queue matmul \
        --config $CFG/stage2_distribution_layers$n.py --tag layers$n > "$LOG/train_layers$n.log" 2>&1
done
python tools/feasibility/infer_bench.py --mode repo --queue matmul --tag matmul_layers6 > "$LOG/infer_layers6.log" 2>&1
for n in 5 4; do
    python tools/feasibility/infer_bench.py --mode repo --queue matmul \
        --config $CFG/stage2_distribution_layers$n.py --tag matmul_layers$n > "$LOG/infer_layers$n.log" 2>&1
done
for n in 5 4; do
    python tools/feasibility/queue_matmul_eval.py --variant matmul --run layers$n --no-save \
        --config $CFG/stage2_distribution_layers$n.py > "$LOG/eval_layers$n.log" 2>&1
done
echo ALL_DONE > "$LOG/done"
