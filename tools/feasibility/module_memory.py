"""Exp 2: per-module inference activation memory of the deploy model (batch 1).

For each top-level stage (img_backbone, img_neck, feature_maps_format,
det_head, map_head, motion_plan_head, post_process) a pre-hook records
memory_allocated and resets the peak counter; the post-hook records
  transient_MB = max_allocated during the call - allocated before
  retained_MB  = allocated after the call - allocated before (output kept alive)
The stages are sequential (never nested), so resetting the peak per stage is safe.
Peak/latency of the whole model are measured separately by infer_bench.py.

  PYTHONPATH=. python tools/feasibility/module_memory.py --mode fp32
"""
import argparse
import os
from collections import defaultdict

import numpy as np
import torch

import common as C

STAGES = ["img_backbone", "img_neck", "feature_maps_format",
          "det_head", "map_head", "motion_plan_head", "post_process"]


class Recorder:
    def __init__(self):
        self.rec = defaultdict(list)
        self.active = False
        self._before = {}

    def pre(self, name):
        if not self.active:
            return
        torch.cuda.synchronize()
        self._before[name] = torch.cuda.memory_allocated()
        torch.cuda.reset_peak_memory_stats()

    def post(self, name):
        if not self.active:
            return
        torch.cuda.synchronize()
        b = self._before.pop(name)
        self.rec[name].append(dict(
            transient_MB=(torch.cuda.max_memory_allocated() - b) / C.MB,
            retained_MB=(torch.cuda.memory_allocated() - b) / C.MB,
        ))


def wrap_callable(owner, attr, name, recorder):
    fn = getattr(owner, attr)

    def wrapped(*args, **kwargs):
        recorder.pre(name)
        out = fn(*args, **kwargs)
        recorder.post(name)
        return out

    setattr(owner, attr, wrapped)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=C.PRECISION_MODES, required=True)
    ap.add_argument("--config", default=C.DEFAULT_CONFIG)
    ap.add_argument("--ckpt", default=C.DEFAULT_CKPT)
    ap.add_argument("--warmup", type=int, default=10)
    ap.add_argument("--iters", type=int, default=100)
    args = ap.parse_args()

    cfg = C.load_cfg(args.config)
    model, _ = C.build_model(cfg, args.ckpt)
    params = {
        "img_backbone": C.param_stats(model.img_backbone),
        "img_neck": C.param_stats(model.img_neck),
        "det_head": C.param_stats(model.head.det_head),
        "map_head": C.param_stats(model.head.map_head),
        "motion_plan_head": C.param_stats(model.head.motion_plan_head),
        "depth_branch (train only)": C.param_stats(model.depth_branch),
        "total": C.param_stats(model),
    }
    C.apply_precision(model, args.mode)
    model.cuda().eval()

    rec = Recorder()
    # module-level hooks (outside of autocast/decorator wrappers, i.e. measured
    # exactly as they run inside the real forward)
    for name in ["img_backbone", "img_neck"]:
        m = getattr(model, name)
        m.register_forward_pre_hook(lambda mod, inp, n=name: rec.pre(n))
        m.register_forward_hook(lambda mod, inp, out, n=name: rec.post(n))
    for name in ["det_head", "map_head", "motion_plan_head"]:
        m = getattr(model.head, name)
        m.register_forward_pre_hook(lambda mod, inp, n=name: rec.pre(n))
        m.register_forward_hook(lambda mod, inp, out, n=name: rec.post(n))
    import projects.mmdet3d_plugin.models.distilldrive as dd_mod
    wrap_callable(dd_mod, "feature_maps_format", "feature_maps_format", rec)
    wrap_callable(model.head, "post_process", "post_process", rec)

    _, loader = C.build_test_loader(cfg)
    samples = C.prefetch(loader, args.warmup + args.iters)
    with torch.no_grad():
        for i, data in enumerate(samples):
            rec.active = i >= args.warmup
            model(return_loss=False, rescale=True, **C.to_gpu(data))

    summary = {}
    for name in STAGES:
        r = rec.rec[name]
        t = np.array([x["transient_MB"] for x in r])
        k = np.array([x["retained_MB"] for x in r])
        summary[name] = dict(n=len(r), transient_MB_mean=float(t.mean()), transient_MB_max=float(t.max()),
                             retained_MB_mean=float(k.mean()), retained_MB_max=float(k.max()))
        print(f"{name:<20} transient mean {t.mean():8.1f} max {t.max():8.1f} | "
              f"retained mean {k.mean():7.1f} max {k.max():7.1f} MB")
    C.dump_json(dict(mode=args.mode, params=params, activations=summary),
                os.path.join(C.OUT_ROOT, f"exp2_module_memory_{args.mode}.json"))


if __name__ == "__main__":
    torch.multiprocessing.set_start_method("fork")
    main()
