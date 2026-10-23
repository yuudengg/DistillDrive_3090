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

  PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:128 PYTHONPATH=. \
      python tools/feasibility/train_memory.py --iters 100
"""
import argparse
import os
import subprocess

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


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=C.DEFAULT_CONFIG)
    ap.add_argument("--ckpt", default=C.DEFAULT_CKPT)
    ap.add_argument("--batch", type=int, default=4)
    ap.add_argument("--iters", type=int, default=100)
    ap.add_argument("--workers", type=int, default=4)
    args = ap.parse_args()
    assert args.iters >= 50

    from mmcv.runner import build_optimizer, wrap_fp16_model
    from mmdet.datasets import build_dataset
    from projects.mmdet3d_plugin.datasets.builder import build_dataloader

    cfg = C.load_cfg(args.config)
    torch.manual_seed(0)
    np.random.seed(0)

    model, _ = C.build_model(cfg, args.ckpt, deploy=False)
    wrap_fp16_model(model)  # what Fp16OptimizerHook.before_run does
    model.cuda().train()

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
        batch = next(it_loader)
        torch.cuda.synchronize()
        a0 = torch.cuda.memory_allocated()
        torch.cuda.reset_peak_memory_stats()
        data = C.to_gpu(batch)
        torch.cuda.synchronize()
        inputs_MB = (torch.cuda.memory_allocated() - a0) / C.MB

        outputs = model.train_step(data, None)
        scaler.scale(outputs["loss"] / accum).backward()
        torch.cuda.synchronize()
        peak_fb = torch.cuda.max_memory_allocated()
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
        rows.append(dict(
            it=it, stepped=stepped,
            static_before_MB=a0 / C.MB,
            inputs_MB=inputs_MB,
            activations_MB=(peak_fb - a0) / C.MB - inputs_MB,
            peak_allocated_MB=torch.cuda.max_memory_allocated() / C.MB,
            peak_reserved_MB=torch.cuda.max_memory_reserved() / C.MB,
            loss=float(outputs["loss"]),
        ))
        r = rows[-1]
        print(f"it {it:3d} static {r['static_before_MB']:7.0f} act {r['activations_MB']:7.0f} "
              f"peak_alloc {r['peak_allocated_MB']:7.0f} peak_res {r['peak_reserved_MB']:7.0f} "
              f"loss {r['loss']:.2f}{' step' if stepped else ''}", flush=True)

    def col(k):
        return np.array([r[k] for r in rows])

    imax = int(col("peak_allocated_MB").argmax())
    report = dict(
        batch=args.batch, iters=args.iters, cumulative_iters=accum,
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
    C.dump_json(report, os.path.join(C.OUT_ROOT, f"exp1_train_memory_bs{args.batch}.json"))


if __name__ == "__main__":
    torch.multiprocessing.set_start_method("fork")
    main()
