"""Run the existing measurement tools unchanged for the Sparse Query experiment, writing into a
separate output folder, and record the Sparse Query head's per-forward sq_stats.

  bench  -> infer_bench.py (latency / peak memory; batch 1, warm-up 10 + 100, repo mode)
  train  -> train_memory.py (batch 4 training-step memory) with the optimizer learning rate forced
            to 0 at every step: AdamW then allocates and updates its state exactly as in training
            (same memory), but every parameter update is p += -0 * (...), i.e. no weight changes.
            A checksum of all optimizer-managed parameters is compared before / after to prove it.
            No checkpoint is written (train_memory.py never saves one).

  PYTHONPATH=. python tools/feasibility/sq_measure.py bench --out work_dirs/sparse_query_eval/bench -- \
      --mode repo --queue matmul --config tools/feasibility/configs/stage2_distribution_det6_map4_sq.py \
      --ckpt work_dirs/recover_stage2_distribution_det6_map4/iter_21096.pth --tag sq64_E_seed0
"""
import json
import os
import sys

import numpy as np
import torch

import common as C

RECORDS = []


def patch_sq_recorder():
    try:
        from projects.mmdet3d_plugin.models.motion.sparse_query_motion_planning_head import \
            SparseQueryMotionPlanningHead as H
    except ImportError:
        return
    orig = H.forward

    def forward(self, *a, **k):
        out = orig(self, *a, **k)
        RECORDS.append({key: int(v) for key, v in self.sq_stats.items() if isinstance(v, (int, np.integer))})
        return out
    H.forward = forward


def patch_zero_lr():
    state = {}
    orig = torch.cuda.amp.GradScaler.step

    def step(self, optimizer, *a, **k):
        params = [p for g in optimizer.param_groups for p in g["params"]]
        if "before" not in state:
            state["before"] = float(sum(p.detach().double().sum() for p in params))
            state["n_params"] = int(sum(p.numel() for p in params))
        for g in optimizer.param_groups:
            g["lr"] = 0.0
        r = orig(self, optimizer, *a, **k)
        state["after"] = float(sum(p.detach().double().sum() for p in params))
        state["steps"] = state.get("steps", 0) + 1
        return r
    torch.cuda.amp.GradScaler.step = step
    return state


def summary(records):
    keys = sorted({k for r in records for k in r})
    out = {k: dict(mean=float(np.mean([r[k] for r in records if k in r])),
                   min=int(min(r[k] for r in records if k in r)),
                   max=int(max(r[k] for r in records if k in r))) for k in keys}
    out["forwards"] = len(records)
    return out


def main():
    mode = sys.argv[1]
    out = sys.argv[sys.argv.index("--out") + 1]
    tool_args = sys.argv[sys.argv.index("--") + 1:]
    os.makedirs(out, exist_ok=True)
    C.OUT_ROOT = os.path.abspath(out)                 # the tools write their JSON under C.OUT_ROOT
    patch_sq_recorder()
    lr_state = patch_zero_lr() if mode == "train" else None
    sys.argv = [mode] + tool_args
    if mode == "bench":
        import infer_bench as T
    elif mode == "train":
        import train_memory as T
    else:
        raise SystemExit(f"unknown mode {mode}")
    T.main()
    extra = dict(mode=mode, tool_args=tool_args, sq_stats=summary(RECORDS) if RECORDS else None)
    if lr_state is not None:
        extra["zero_lr"] = dict(lr_state, unchanged=lr_state.get("before") == lr_state.get("after"))
    tag = tool_args[tool_args.index("--tag") + 1] if "--tag" in tool_args else mode
    with open(os.path.join(out, f"sq_measure_{mode}_{tag}.json"), "w") as f:
        json.dump(extra, f, indent=1)
    print("sq_measure:", json.dumps({k: v for k, v in extra.items() if k != "tool_args"}))


if __name__ == "__main__":
    torch.multiprocessing.set_start_method("fork")
    main()
