"""Exp 3: deploy-model inference latency/memory per precision mode (+ optional full val eval).

Latency protocol: batch 1, 10 warmup + 100 timed consecutive val frames,
torch.cuda.synchronize() around each forward; H2D copy of the frame is not timed.

  PYTHONPATH=. python tools/feasibility/infer_bench.py --mode full_amp --eval
"""
import argparse
import os
import time

import numpy as np
import torch

import common as C


def bench(model, samples, n_warmup):
    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()
    static_alloc = torch.cuda.memory_allocated()
    times = []
    with torch.no_grad():
        for i, data in enumerate(samples):
            data = C.to_gpu(data)
            torch.cuda.synchronize()
            t0 = time.perf_counter()
            model(return_loss=False, rescale=True, **data)
            torch.cuda.synchronize()
            if i >= n_warmup:
                times.append((time.perf_counter() - t0) * 1000)
    times = np.array(times)
    return dict(
        n_timed=len(times),
        latency_ms_mean=float(times.mean()),
        latency_ms_std=float(times.std()),
        fps=float(1000.0 / times.mean()),
        latency_ms_median=float(np.median(times)),
        latency_ms_p90=float(np.percentile(times, 90)),
        model_static_alloc_MB=static_alloc / C.MB,
        peak_allocated_MB=torch.cuda.max_memory_allocated() / C.MB,
        peak_reserved_MB=torch.cuda.max_memory_reserved() / C.MB,
    )


def run_inference(model, loader, limit=None):
    import mmcv

    results = []
    prog = mmcv.ProgressBar(limit or len(loader.dataset))
    with torch.no_grad():
        for i, data in enumerate(loader):
            if limit is not None and i >= limit:
                break
            results.extend(model(return_loss=False, rescale=True, **C.to_gpu(data)))
            prog.update()
    return results


def evaluate(cfg, dataset, results):
    eval_kwargs = dict(cfg.get("evaluation", {}))
    for k in ["interval", "tmpdir", "start", "gpu_collect", "save_best", "rule"]:
        eval_kwargs.pop(k, None)
    metrics = dataset.evaluate(results, metric="bbox", **eval_kwargs)
    keep = {
        "L2": "L2", "obj_box_col": "obj_box_col", "NDS": "img_bbox_NuScenes/NDS",
        "det_mAP": "img_bbox_NuScenes/mAP", "map_mAP": "mAP_normal",
        "obj_col": "obj_col", "car_EPA": "car_EPA", "pedestrian_EPA": "pedestrian_EPA",
    }
    summary = {name: float(metrics[k]) for name, k in keep.items() if k in metrics}
    return summary, {k: (float(v) if isinstance(v, (int, float, np.floating)) else str(v))
                     for k, v in metrics.items()}


def full_eval(cfg, mode, out_dir):
    model, _ = C.build_model(cfg)
    C.apply_precision(model, mode)
    model.cuda().eval()
    dataset, loader = C.build_test_loader(cfg, work_dir=out_dir)
    return evaluate(cfg, dataset, run_inference(model, loader))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=C.PRECISION_MODES, required=True)
    ap.add_argument("--config", default=C.DEFAULT_CONFIG)
    ap.add_argument("--ckpt", default=C.DEFAULT_CKPT)
    ap.add_argument("--warmup", type=int, default=10)
    ap.add_argument("--iters", type=int, default=100)
    ap.add_argument("--eval", action="store_true", help="also run full val evaluation")
    ap.add_argument("--skip-bench", action="store_true")
    ap.add_argument("--queue", choices=["original", "matmul"], default="original")
    ap.add_argument("--tag", default="", help="suffix of the output file name")
    args = ap.parse_args()

    cfg = C.load_cfg(args.config)
    C.set_queue_variant(args.queue)
    report = dict(mode=args.mode, config=args.config, ckpt=args.ckpt, queue=args.queue,
                  deploy="distillation_head excluded (with_distillation=False)",
                  cudnn_benchmark=torch.backends.cudnn.benchmark)

    if not args.skip_bench:
        model, n_dropped = C.build_model(cfg, args.ckpt)
        report["dropped_teacher_keys"] = n_dropped
        report["model_params"] = C.param_stats(model)
        C.apply_precision(model, args.mode)
        model.cuda().eval()
        _, loader = C.build_test_loader(cfg)
        samples = C.prefetch(loader, args.warmup + args.iters)
        report["bench"] = bench(model, samples, args.warmup)
        print(report["bench"])
        del model, samples
        torch.cuda.empty_cache()

    if args.eval:
        out_dir = os.path.join(C.OUT_ROOT, f"eval_{args.mode}")
        os.makedirs(out_dir, exist_ok=True)
        summary, full = full_eval(cfg, args.mode, out_dir)
        report["eval"] = summary
        C.dump_json(full, os.path.join(out_dir, "metrics_full.json"))
        print(summary)

    suffix = "_eval" if args.eval and args.skip_bench else ""
    tag = f"_{args.tag}" if args.tag else ""
    C.dump_json(report, os.path.join(C.OUT_ROOT, f"exp3_infer_{args.mode}{tag}{suffix}.json"))


if __name__ == "__main__":
    torch.multiprocessing.set_start_method("fork")
    main()
