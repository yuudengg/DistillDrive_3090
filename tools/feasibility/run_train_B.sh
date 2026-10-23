#!/usr/bin/env bash
# Exp B: training memory per module and with_cp on/off (batch 4, 60 iters each).
# *_hooks runs give the per-module memory, *_timing runs (no hook syncs) the iter time.
set -eo pipefail
cd "$(dirname "$0")/../.."
source .venv/bin/activate
export PYTHONPATH="$(pwd)${PYTHONPATH:+:$PYTHONPATH}"
export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:128   # same as the real stage2 run
LOG=tools/feasibility/outputs/logs_B
mkdir -p "$LOG"
for cp in on off; do
    python tools/feasibility/train_memory.py --iters 60 --with-cp "$cp" --hooks > "$LOG/cp${cp}_hooks.log" 2>&1
    python tools/feasibility/train_memory.py --iters 60 --with-cp "$cp" > "$LOG/cp${cp}_timing.log" 2>&1
done
rm -rf tools/feasibility/outputs/train_tmp
echo ALL_DONE > "$LOG/done"
