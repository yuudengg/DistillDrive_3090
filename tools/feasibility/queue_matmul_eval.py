"""Exp A: full val evaluation of InstanceQueue.prepare_motion original vs matmul (repo mode).

Saves the raw per-sample outputs (like tools/test.py --out) for compare_outputs.py,
and records the TF32 / cudnn flags and statistics of the tracking match matrix
(dtype, distinct values, max matches per row/column) seen during the run.

  PYTHONPATH=. python tools/feasibility/queue_matmul_eval.py --variant matmul --run matmul
  PYTHONPATH=. python tools/feasibility/queue_matmul_eval.py --variant original --run original_2 --no-metrics
"""
import argparse
import os
import random
import shutil
import time

import mmcv
import numpy as np
import torch

import common as C
from infer_bench import evaluate, run_inference


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--variant", choices=["original", "matmul"], required=True)
    ap.add_argument("--run", required=True, help="output sub-directory name")
    ap.add_argument("--mode", choices=C.PRECISION_MODES, default="repo")
    ap.add_argument("--config", default=C.DEFAULT_CONFIG)
    ap.add_argument("--no-save", action="store_true", help="do not keep outputs.pkl")
    ap.add_argument("--eval-layer", type=int, default=None,
                    help="diagnostic: det/map results for evaluation are read from this decoder "
                         "layer (1-based); the model still runs all layers and the cache, instance "
                         "ids, motion_plan_head and its post-processing use the last layer")
    ap.add_argument("--no-metrics", action="store_true", help="inference + --out only")
    ap.add_argument("--limit", type=int, default=None, help="first N frames only (smoke test)")
    ap.add_argument("--seed", type=int, default=0,
                    help="motion_plan_head samples torch.randn_like(present_mu) at test time")
    args = ap.parse_args()
    assert args.limit is None or args.no_metrics, "--limit needs --no-metrics"

    cfg = C.load_cfg(args.config)
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    out_dir = os.path.join(C.OUT_ROOT, "queue_eval", args.run)
    os.makedirs(out_dir, exist_ok=True)

    model, _ = C.build_model(cfg)
    C.apply_precision(model, args.mode)
    model.cuda().eval()
    match_stats = {}
    C.set_queue_variant(args.variant, match_stats)
    if args.eval_layer is not None:
        C.read_eval_layer(model, args.eval_layer)

    report = dict(
        variant=args.variant, mode=args.mode, seed=args.seed, config=args.config,
        eval_layer=args.eval_layer,
        flags=dict(
            cuda_matmul_allow_tf32=torch.backends.cuda.matmul.allow_tf32,
            cudnn_allow_tf32=torch.backends.cudnn.allow_tf32,
            cudnn_benchmark=torch.backends.cudnn.benchmark,
            cudnn_deterministic=torch.backends.cudnn.deterministic,
        ),
    )
    dataset, loader = C.build_test_loader(cfg, work_dir=out_dir)
    t0 = time.time()
    results = run_inference(model, loader, args.limit)
    report["inference_min"] = (time.time() - t0) / 60
    if not args.no_save:
        out_pkl = os.path.join(out_dir, "outputs.pkl")
        mmcv.dump(results, out_pkl)
        report["out"] = out_pkl
    report["match"] = dict(match_stats, dtypes=sorted(match_stats["dtypes"]),
                           values=sorted(match_stats["values"]))
    print(report["match"])

    if not args.no_metrics:
        summary, full = evaluate(cfg, dataset, results)
        report["eval"] = summary
        C.dump_json(full, os.path.join(out_dir, "metrics_full.json"))
        print(summary)
        # dataset.evaluate's own copy of the outputs + format intermediates
        for f in ["results.pkl", "results_nusc.json", "submission_vector.json", "metrics_details.json"]:
            if os.path.exists(os.path.join(out_dir, f)):
                os.remove(os.path.join(out_dir, f))
        shutil.rmtree(os.path.join(out_dir, "plots"), ignore_errors=True)
    C.dump_json(report, os.path.join(out_dir, "report.json"))


if __name__ == "__main__":
    torch.multiprocessing.set_start_method("fork")
    main()
