"""Exp 1: training-memory breakdown of stage2 (full model incl. frozen teacher).

Mirrors the real stage2 run: train pipeline + GroupInBatchSampler, batch 4,
wrap_fp16_model + GradScaler(loss_scale=32), AdamW from cfg.optimizer,
cumulative_iters from cfg.optimizer_config, grad clip 25. No checkpoint is saved;
the weights are only used to get realistic instance counts.

Buckets
  weights     : all parameters + buffers (trainable student / frozen teacher split)
  grads       : p.grad of trainable params (kept allocated: zero_grad(set_to_none=False))
  optimizer   : AdamW exp_avg / exp_avg_sq (+ step) after the first optimizer step
  inputs      : the scattered batch on GPU
  activations : per-iter peak during forward+backward - allocated before the iter - inputs
Each iteration's peak is recorded and the max over all iterations is reported.

Exp B options
  --with-cp on|off : override img_backbone.with_cp (config value is True)
  --hooks heads    : det_head / map_head split per decoder layer and op (see head_breakdown.py)
  --hooks [modules]: per-module training memory. For img_backbone, img_neck,
                     depth_branch, distillation_head, det_head, map_head,
                     motion_plan_head (forward hooks), head.loss and backward:
                       retained  = allocated after - before (kept for backward + outputs)
                       transient = peak during - allocated before
                     Peak counters are reset per segment, but a running max keeps the
                     whole-iteration peak exact. Hook syncs perturb timing, so iter
                     times are taken from a run without --hooks.
  --queue matmul   : InstanceQueue.prepare_motion with the broadcast-sum as matmul
  step_ms (data already on GPU -> optimizer done, synchronized) and data_wait_ms
  are recorded per iteration.

  PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:128 PYTHONPATH=. \
      python tools/feasibility/train_memory.py --iters 100
"""
import argparse
import os
import subprocess
import time
from collections import defaultdict

import numpy as np
import torch

import common as C


def gpu_process_used_MB():
    try:
        out = subprocess.check_output(
            ["nvidia-smi", "--query-compute-apps=pid,used_memory", "--format=csv,noheader,nounits"],
            text=True)
        for line in out.strip().splitlines():
            pid, used = [x.strip() for x in line.split(",")]
            if int(pid) == os.getpid():
                return float(used)
    except Exception:
        pass
    return None


SEGMENTS = ["img_backbone", "img_neck", "depth_branch", "distillation_head",
            "det_head", "map_head", "motion_plan_head", "head.loss", "backward"]


class SegmentMemory:
    """Per-segment allocated/peak tracking that keeps an exact running iteration peak.
    Segments may nest: a peak seen inside an inner segment is folded into every
    open (outer) segment before the counter is reset."""

    def __init__(self):
        self.rows = defaultdict(list)
        self.iter_peak = 0
        self.it = -1
        self._open = {}

    def _fold_peak(self):
        m = torch.cuda.max_memory_allocated()
        self.iter_peak = max(self.iter_peak, m)
        for v in self._open.values():
            v[1] = max(v[1], m)
        torch.cuda.reset_peak_memory_stats()

    def start_iter(self):
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()
        self.iter_peak = torch.cuda.memory_allocated()
        self.it += 1

    def pre(self, name):
        torch.cuda.synchronize()
        self._fold_peak()
        a = torch.cuda.memory_allocated()
        self._open[name] = [a, a]

    def post(self, name):
        torch.cuda.synchronize()
        self._fold_peak()
        before, peak = self._open.pop(name)
        self.rows[name].append(dict(it=self.it, transient_MB=(peak - before) / C.MB,
                                    retained_MB=(torch.cuda.memory_allocated() - before) / C.MB))

    def end_iter(self):
        torch.cuda.synchronize()
        self._fold_peak()
        return self.iter_peak

    def summary(self):
        """Per name: retained summed over calls within an iteration, transient = max
        over calls; then mean / max over iterations."""
        out = {}
        for name, rr in self.rows.items():
            per_it = defaultdict(lambda: [0.0, 0.0, 0])
            for x in rr:
                v = per_it[x["it"]]
                v[0] += x["retained_MB"]
                v[1] = max(v[1], x["transient_MB"])
                v[2] += 1
            k = np.array([v[0] for v in per_it.values()])
            t = np.array([v[1] for v in per_it.values()])
            out[name] = dict(n_iters=len(per_it), calls_per_iter=max(v[2] for v in per_it.values()),
                             retained_MB_mean=float(k.mean()), retained_MB_max=float(k.max()),
                             transient_MB_mean=float(t.mean()), transient_MB_max=float(t.max()))
        return out


def install_hooks(model, seg):
    mods = dict(img_backbone=model.img_backbone, img_neck=model.img_neck,
                depth_branch=model.depth_branch,
                distillation_head=model.head.distillation_head,
                det_head=model.head.det_head, map_head=model.head.map_head,
                motion_plan_head=model.head.motion_plan_head)
    for name, m in mods.items():
        m.register_forward_pre_hook(lambda mod, inp, n=name: seg.pre(n))
        m.register_forward_hook(lambda mod, inp, out, n=name: seg.post(n))
    loss_fn = model.head.loss

    def loss_wrapped(*a, **k):
        seg.pre("head.loss")
        out = loss_fn(*a, **k)
        seg.post("head.loss")
        return out

    model.head.loss = loss_wrapped


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=C.DEFAULT_CONFIG)
    ap.add_argument("--ckpt", default=C.DEFAULT_CKPT)
    ap.add_argument("--batch", type=int, default=4)
    ap.add_argument("--iters", type=int, default=100)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--with-cp", choices=["cfg", "on", "off"], default="cfg")
    ap.add_argument("--hooks", nargs="?", const="modules", choices=["modules", "heads"], default=None,
                    help="modules: top-level modules (Exp B); heads: det/map head per layer and op "
                         "(head_breakdown.install_head_wrappers)")
    ap.add_argument("--tag", default="", help="suffix of the output file name")
    ap.add_argument("--queue", choices=["original", "matmul"], default="original",
                    help="InstanceQueue.prepare_motion implementation (common.set_queue_variant)")
    args = ap.parse_args()
    if args.iters < 50:
        print("WARNING: fewer than 50 iterations (smoke run)")

    from mmcv.runner import build_optimizer, wrap_fp16_model
    from mmdet.datasets import build_dataset
    from projects.mmdet3d_plugin.datasets.builder import build_dataloader

    cfg = C.load_cfg(args.config)
    if args.with_cp != "cfg":
        cfg.model.img_backbone.with_cp = args.with_cp == "on"
    torch.manual_seed(0)
    np.random.seed(0)

    model, _ = C.build_model(cfg, args.ckpt, deploy=False)
    wrap_fp16_model(model)  # what Fp16OptimizerHook.before_run does
    model.cuda().train()
    C.set_queue_variant(args.queue)
    seg = SegmentMemory() if args.hooks else None
    if args.hooks == "modules":
        install_hooks(model, seg)
    elif args.hooks == "heads":
        from head_breakdown import install_head_wrappers
        install_head_wrappers(model, seg)

    teacher = model.head.distillation_head
    teacher_ids = {id(p) for p in teacher.parameters()}
    params = list(model.parameters())
    trainable = [p for p in params if p.requires_grad]
    w = dict(
        student_trainable_MB=C.tensor_bytes(trainable) / C.MB,
        teacher_frozen_MB=C.tensor_bytes([p for p in params if id(p) in teacher_ids]) / C.MB,
        other_frozen_MB=C.tensor_bytes([p for p in params if not p.requires_grad and id(p) not in teacher_ids]) / C.MB,
        buffers_MB=C.tensor_bytes(list(model.buffers())) / C.MB,
        trainable_params_M=sum(p.numel() for p in trainable) / 1e6,
        teacher_params_M=sum(p.numel() for p in teacher.parameters()) / 1e6,
    )
    w["total_MB"] = w["student_trainable_MB"] + w["teacher_frozen_MB"] + w["other_frozen_MB"] + w["buffers_MB"]

    optimizer = build_optimizer(model, cfg.optimizer)
    loss_scale = cfg.fp16.get("loss_scale", 512.0)
    scaler = torch.cuda.amp.GradScaler(init_scale=loss_scale)
    accum = cfg.optimizer_config.get("cumulative_iters", 1)
    max_norm = cfg.optimizer_config.grad_clip.max_norm

    cfg.data.train.work_dir = os.path.join(C.OUT_ROOT, "train_tmp")
    dataset = build_dataset(cfg.data.train)
    loader = build_dataloader(dataset, args.batch, args.workers, num_gpus=1, dist=False,
                              seed=0, runner_type="IterBasedRunner")
    it_loader = iter(loader)

    torch.cuda.synchronize()
    base_after_weights = torch.cuda.memory_allocated()
    grads_MB = opt_MB = None
    rows = []
    for it in range(args.iters):
        t_data = time.perf_counter()
        batch = next(it_loader)
        data_wait_ms = (time.perf_counter() - t_data) * 1000
        torch.cuda.synchronize()
        a0 = torch.cuda.memory_allocated()
        torch.cuda.reset_peak_memory_stats()
        data = C.to_gpu(batch)
        torch.cuda.synchronize()
        inputs_MB = (torch.cuda.memory_allocated() - a0) / C.MB
        if seg is not None:
            seg.start_iter()

        t0 = time.perf_counter()
        outputs = model.train_step(data, None)
        if seg is not None:
            seg.pre("backward")
        scaler.scale(outputs["loss"] / accum).backward()
        if seg is not None:
            seg.post("backward")
        torch.cuda.synchronize()
        peak_fb = seg.end_iter() if seg is not None else torch.cuda.max_memory_allocated()
        if grads_MB is None:
            grads_MB = C.tensor_bytes([p.grad for p in trainable if p.grad is not None]) / C.MB

        stepped = (it + 1) % accum == 0
        if stepped:
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_([p for p in trainable if p.grad is not None], max_norm)
            scaler.step(optimizer)
            scaler.update(loss_scale)
            model.zero_grad()
            optimizer.zero_grad()
            if opt_MB is None:
                opt_MB = C.tensor_bytes([t for s in optimizer.state.values()
                                         for t in s.values() if torch.is_tensor(t) and t.is_cuda]) / C.MB
        torch.cuda.synchronize()
        step_ms = (time.perf_counter() - t0) * 1000
        peak_iter = max(peak_fb, torch.cuda.max_memory_allocated())
        rows.append(dict(
            it=it, stepped=stepped,
            static_before_MB=a0 / C.MB,
            inputs_MB=inputs_MB,
            activations_MB=(peak_fb - a0) / C.MB - inputs_MB,
            peak_allocated_MB=peak_iter / C.MB,
            peak_reserved_MB=torch.cuda.max_memory_reserved() / C.MB,
            step_ms=step_ms,
            data_wait_ms=data_wait_ms,
            loss=float(outputs["loss"]),
        ))
        r = rows[-1]
        print(f"it {it:3d} static {r['static_before_MB']:7.0f} act {r['activations_MB']:7.0f} "
              f"peak_alloc {r['peak_allocated_MB']:7.0f} peak_res {r['peak_reserved_MB']:7.0f} "
              f"step {step_ms:6.0f}ms data {data_wait_ms:5.0f}ms "
              f"loss {r['loss']:.2f}{' step' if stepped else ''}", flush=True)

    def col(k):
        return np.array([r[k] for r in rows])

    imax = int(col("peak_allocated_MB").argmax())
    warm = 10  # first iterations excluded from timing (cudnn/allocator warmup)
    steady = [r for r in rows[warm:]]
    sm = np.array([r["step_ms"] for r in steady])
    timing = dict(
        excluded_warmup_iters=warm,
        step_ms_mean=float(sm.mean()), step_ms_median=float(np.median(sm)),
        step_ms_mean_no_opt_step=float(np.mean([r["step_ms"] for r in steady if not r["stepped"]])),
        data_wait_ms_mean=float(np.mean([r["data_wait_ms"] for r in steady])),
        iter_ms_mean=float(np.mean([r["step_ms"] + r["data_wait_ms"] for r in steady])),
    )
    modules = None
    if seg is not None:
        modules = seg.summary()
        if args.hooks == "modules":
            for name in SEGMENTS:
                m = modules[name]
                print(f"{name:<18} retained mean {m['retained_MB_mean']:8.1f} max {m['retained_MB_max']:8.1f} | "
                      f"transient mean {m['transient_MB_mean']:8.1f} max {m['transient_MB_max']:8.1f} MB")
    report = dict(
        batch=args.batch, iters=args.iters, cumulative_iters=accum,
        with_cp=bool(model.img_backbone.with_cp), hooks=args.hooks, queue=args.queue, config=args.config,
        timing=timing, modules=modules,
        alloc_conf=os.environ.get("PYTORCH_CUDA_ALLOC_CONF"),
        weights=w,
        weights_alloc_measured_MB=base_after_weights / C.MB,
        grads_MB=grads_MB,
        optimizer_state_MB=opt_MB,
        inputs_MB_max=float(col("inputs_MB").max()),
        activations_MB=dict(max=float(col("activations_MB").max()), mean=float(col("activations_MB").mean()),
                            p50=float(np.median(col("activations_MB")))),
        peak_allocated_MB_max=float(col("peak_allocated_MB").max()),
        peak_allocated_iter=imax,
        peak_reserved_MB_max=float(col("peak_reserved_MB").max()),
        nvidia_smi_process_MB_end=gpu_process_used_MB(),
        gpu_total_MB=torch.cuda.get_device_properties(0).total_memory / C.MB,
        per_iter=rows,
    )
    if args.with_cp == "cfg" and not args.hooks and args.queue == "original":
        name = f"exp1_train_memory_bs{args.batch}.json"
    else:
        name = f"expB_train_bs{args.batch}_cp{'on' if model.img_backbone.with_cp else 'off'}" \
               f"_{ {'modules': 'hooks', 'heads': 'heads', None: 'timing'}[args.hooks] }" \
               f"{'_matmul' if args.queue == 'matmul' else ''}{'_' + args.tag if args.tag else ''}.json"
    C.dump_json(report, os.path.join(C.OUT_ROOT, name))


if __name__ == "__main__":
    torch.multiprocessing.set_start_method("fork")
    main()
