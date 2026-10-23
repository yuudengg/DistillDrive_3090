#!/usr/bin/env bash
# Exp 2 + Exp 3 for every precision mode, then full val eval, then the queue probe.
# Run from anywhere:  bash tools/feasibility/run_infer_all.sh
set -eo pipefail
cd "$(dirname "$0")/../.."
source .venv/bin/activate
export PYTHONPATH="$(pwd)${PYTHONPATH:+:$PYTHONPATH}"
LOG=tools/feasibility/outputs/logs
mkdir -p "$LOG"

for m in fp32 repo full_amp; do
    python tools/feasibility/module_memory.py --mode "$m" > "$LOG/exp2_$m.log" 2>&1
    python tools/feasibility/infer_bench.py --mode "$m" > "$LOG/exp3_bench_$m.log" 2>&1
done
python tools/feasibility/infer_bench.py --mode full_amp_raw > "$LOG/exp3_bench_full_amp_raw.log" 2>&1
python tools/feasibility/probe_motion_head.py --mode fp32 > "$LOG/probe_motion_head.log" 2>&1
python tools/feasibility/probe_fp16_nan.py --mode full_amp_raw --frames 440 > "$LOG/probe_fp16_nan_raw.log" 2>&1
python tools/feasibility/probe_fp16_nan.py --mode full_amp --frames 1000 > "$LOG/probe_fp16_nan_full_amp.log" 2>&1

# full_amp = FP16 candidate; repo = what tools/test.py runs (source of the reference
# metrics); fp32 = true fp32. Large intermediates (~4GB per eval) are removed afterwards;
# metrics_full.json / metrics_summary.json keep every metric.
for m in full_amp repo fp32; do
    python tools/feasibility/infer_bench.py --mode "$m" --skip-bench --eval > "$LOG/exp3_eval_$m.log" 2>&1
    d=tools/feasibility/outputs/eval_$m
    rm -rf "$d/results.pkl" "$d/results_nusc.json" "$d/submission_vector.json" "$d/metrics_details.json" "$d/plots"
done
for m in fp32 full_amp; do
    python tools/feasibility/probe_queue_matmul.py --mode "$m" > "$LOG/probe_queue_matmul_$m.log" 2>&1
done
echo ALL_DONE > "$LOG/run_infer_all.done"
