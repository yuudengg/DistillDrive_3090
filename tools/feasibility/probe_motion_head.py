"""Exp 2 follow-up: which ops make motion_plan_head's ~800MB transient peak,
and which dtype DeformableAggregation actually runs in under each precision mode.

  PYTHONPATH=. python tools/feasibility/probe_motion_head.py --mode full_amp
"""
import argparse
import os
from collections import Counter

import torch
from torch.profiler import ProfilerActivity, profile, record_function

import common as C


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=C.PRECISION_MODES, default="full_amp")
    ap.add_argument("--warmup", type=int, default=10)
    args = ap.parse_args()

    cfg = C.load_cfg()
    model, _ = C.build_model(cfg)
    C.apply_precision(model, args.mode)
    model.cuda().eval()

    # dtype seen by the custom CUDA op (its wrapper casts to float before the kernel)
    import projects.mmdet3d_plugin.ops.deformable_aggregation as daf
    seen = Counter()
    orig_fwd = daf.DeformableAggregationFunction.forward

    def spy(ctx, mc_ms_feat, spatial_shape, scale_start_index, sampling_location, weights):
        seen[(str(mc_ms_feat.dtype), str(sampling_location.dtype), str(weights.dtype))] += 1
        out = orig_fwd(ctx, mc_ms_feat, spatial_shape, scale_start_index, sampling_location, weights)
        seen[("kernel_out", str(out.dtype))] += 1
        return out

    daf.DeformableAggregationFunction.forward = staticmethod(spy)

    mph = model.head.motion_plan_head
    for name, m in mph.named_children():
        fwd = m.forward

        def wrapped(*a, _f=fwd, _n=name, **k):
            with record_function(f"mph.{_n}"):
                return _f(*a, **k)

        m.forward = wrapped

    _, loader = C.build_test_loader(cfg)
    samples = C.prefetch(loader, args.warmup + 1)
    with torch.no_grad():
        for d in samples[:-1]:
            model(return_loss=False, rescale=True, **C.to_gpu(d))
        seen.clear()
        data = C.to_gpu(samples[-1])
        torch.cuda.synchronize()
        with profile(activities=[ProfilerActivity.CUDA, ProfilerActivity.CPU],
                     profile_memory=True, record_shapes=True) as prof:
            model(return_loss=False, rescale=True, **data)
            torch.cuda.synchronize()

    table = prof.key_averages(group_by_input_shape=True).table(
        sort_by="self_cuda_memory_usage", row_limit=25)
    print(table)
    print("DeformableAggregation dtypes (in: feat, loc, weights):", dict(seen))
    out = os.path.join(C.OUT_ROOT, f"probe_motion_head_{args.mode}.txt")
    with open(out, "w") as f:
        f.write(table + "\n\nDeformableAggregation dtypes: " + repr(dict(seen)) + "\n")
    print(f"[saved] {out}")


if __name__ == "__main__":
    torch.multiprocessing.set_start_method("fork")
    main()
