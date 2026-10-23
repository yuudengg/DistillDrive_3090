"""Step 2: does the matmul form of InstanceQueue.prepare_motion give the same gradients?

1. Real training state (train mode, train pipeline, batch 4, wrap_fp16_model,
   forward + backward like train_memory.py) for --iters iterations with the
   original prepare_motion. Every call's real inputs (tracking match, each
   instance_feature_queue / anchor_queue entry) are copied to CPU together with
   what autograd sees during training (requires_grad, grad_fn, dtype, autocast).
2. With the training model freed, for every captured input the queue tensor is
   made a leaf with requires_grad=True and both expressions are back-propagated
   with the same upstream gradients (3 seeded randn + ones):
       original : (match[..., None] * q[:, None]).sum(dim=2)
       matmul   : torch.matmul(match.to(q.dtype), q)        (autocast off)
   q.grad of each is compared with the other and with the analytic match^T @ g.
3. Gradient flow through match: bool -> float conversion is checked for a
   grad_fn / requires_grad; additionally, with a float match forced to
   requires_grad=True, the two expressions' match-gradients are compared.

  PYTHONPATH=. python tools/feasibility/probe_queue_grad.py --iters 10
"""
import argparse
import os

import numpy as np
import torch

import common as C


def capture_training_inputs(cfg, n_iters, batch, workers):
    from mmcv.runner import wrap_fp16_model
    from mmdet.datasets import build_dataset
    from projects.mmdet3d_plugin.datasets.builder import build_dataloader
    from projects.mmdet3d_plugin.models.motion.instance_queue import InstanceQueue

    torch.manual_seed(0)
    np.random.seed(0)
    model, _ = C.build_model(cfg, deploy=False)
    wrap_fp16_model(model)
    model.cuda().train()
    C.set_queue_variant("original")
    original = InstanceQueue.prepare_motion
    captured, seen = [], []

    def capture(self, det_output, mask):
        if self.period is not None:
            match = C._tracking_match(self, det_output)
            mf = match.to(torch.float32)
            entry = dict(match=match.detach().cpu(), queues=[])
            seen.append(dict(
                match_dtype=str(match.dtype), match_requires_grad=bool(match.requires_grad),
                match_float_requires_grad=bool(mf.requires_grad), match_float_grad_fn=str(mf.grad_fn),
                autocast_enabled=torch.is_autocast_enabled(),
                det_feature_requires_grad=bool(det_output["instance_feature"].requires_grad),
                queues=[dict(kind=k, dtype=str(q.dtype), requires_grad=bool(q.requires_grad),
                             grad_fn=str(q.grad_fn), is_leaf=bool(q.is_leaf))
                        for k, qs in (("feature", self.instance_feature_queue), ("anchor", self.anchor_queue))
                        for q in qs],
            ))
            for kind, qs in (("feature", self.instance_feature_queue), ("anchor", self.anchor_queue)):
                for q in qs:
                    entry["queues"].append((kind, q.detach().cpu()))
            captured.append(entry)
        return original(self, det_output, mask)

    InstanceQueue.prepare_motion = capture
    cfg.data.train.work_dir = os.path.join(C.OUT_ROOT, "train_tmp")
    loader = build_dataloader(build_dataset(cfg.data.train), batch, workers, num_gpus=1,
                              dist=False, seed=0, runner_type="IterBasedRunner")
    it = iter(loader)
    for _ in range(n_iters):
        out = model.train_step(C.to_gpu(next(it)), None)
        out["loss"].backward()
        model.zero_grad()
    InstanceQueue.prepare_motion = original
    del model
    torch.cuda.empty_cache()
    return captured, seen


def grad_check(match, q, upstream):
    """q.grad from both expressions for the same upstream gradient g."""
    q_ref = q.clone().requires_grad_(True)
    ref = (match[..., None] * q_ref[:, None]).sum(dim=2)
    ref.backward(upstream)
    q_new = q.clone().requires_grad_(True)
    with torch.autocast(device_type="cuda", enabled=False):
        new = torch.matmul(match.to(q.dtype), q_new)
    new.backward(upstream)
    analytic = torch.matmul(match.to(q.dtype).transpose(-1, -2), upstream)
    return dict(
        fwd_max_abs=float((ref - new).abs().max()),
        grad_max_abs=float((q_ref.grad - q_new.grad).abs().max()),
        grad_ref_vs_analytic=float((q_ref.grad - analytic).abs().max()),
        grad_new_vs_analytic=float((q_new.grad - analytic).abs().max()),
        grad_absmax=float(analytic.abs().max()),
        new_graph_reaches_match=_graph_has_leaf(new.grad_fn, match),
    )


def _graph_has_leaf(fn, tensor, depth=0):
    """True if an AccumulateGrad for `tensor` is in the autograd graph below fn."""
    if fn is None or depth > 50:
        return False
    if getattr(fn, "variable", None) is tensor:
        return True
    return any(_graph_has_leaf(nf, tensor, depth + 1) for nf, _ in fn.next_functions)


def float_match_grad(match, q, upstream):
    """Hypothetical: if match were a float tensor requiring grad, do both forms agree?"""
    out = {}
    m_ref = match.to(q.dtype).requires_grad_(True)
    (m_ref[..., None] * q[:, None]).sum(dim=2).backward(upstream)
    m_new = match.to(q.dtype).requires_grad_(True)
    with torch.autocast(device_type="cuda", enabled=False):
        torch.matmul(m_new, q).backward(upstream)
    out["match_grad_max_abs"] = float((m_ref.grad - m_new.grad).abs().max())
    out["match_grad_absmax"] = float(m_ref.grad.abs().max())
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--iters", type=int, default=10)
    ap.add_argument("--batch", type=int, default=4)
    ap.add_argument("--workers", type=int, default=4)
    args = ap.parse_args()

    cfg = C.load_cfg()
    captured, seen = capture_training_inputs(cfg, args.iters, args.batch, args.workers)
    print(f"captured {len(captured)} prepare_motion calls, "
          f"{sum(len(c['queues']) for c in captured)} queue tensors")

    rows = []
    gens = [("randn_seed0", 0), ("randn_seed1", 1), ("randn_seed2", 2), ("ones", None)]
    for ci, c in enumerate(captured):
        match = c["match"].cuda()
        for qi, (kind, q) in enumerate(c["queues"]):
            q = q.cuda().float()
            for gname, seed in gens:
                if seed is None:
                    g = torch.ones(match.shape[:2] + q.shape[-1:], device="cuda")
                else:
                    gen = torch.Generator(device="cuda").manual_seed(seed)
                    g = torch.randn(match.shape[:2] + q.shape[-1:], device="cuda", generator=gen)
                r = grad_check(match, q, g)
                r.update(call=ci, queue_index=qi, kind=kind, upstream=gname,
                         n_matched=int(match.sum()), max_per_col=int(match.sum(-2).max()))
                rows.append(r)
        # hypothetical float-match gradient, once per call on the first feature entry
        qf = c["queues"][0][1].cuda().float()
        gf = torch.randn(match.shape[:2] + qf.shape[-1:], device="cuda",
                         generator=torch.Generator(device="cuda").manual_seed(0))
        rows[-1].update(float_match_grad(match, qf, gf))
        del match
        torch.cuda.empty_cache()

    def mx(key, kind=None):
        v = [r[key] for r in rows if key in r and (kind is None or r["kind"] == kind)]
        return max(v) if v else None

    summary = dict(
        n_calls=len(captured), n_checks=len(rows),
        upstreams=[g for g, _ in gens],
        grad_max_abs_feature=mx("grad_max_abs", "feature"),
        grad_max_abs_anchor=mx("grad_max_abs", "anchor"),
        grad_ref_vs_analytic=mx("grad_ref_vs_analytic"),
        grad_new_vs_analytic=mx("grad_new_vs_analytic"),
        fwd_max_abs=mx("fwd_max_abs"),
        grad_absmax=mx("grad_absmax"),
        matmul_graph_reaches_match=any(r["new_graph_reaches_match"] for r in rows),
        max_matches_per_column=mx("max_per_col"),
        hypothetical_float_match_grad_max_abs=mx("match_grad_max_abs"),
        hypothetical_float_match_grad_absmax=mx("match_grad_absmax"),
        training_autograd_state=dict(
            match_dtypes=sorted({s["match_dtype"] for s in seen}),
            match_requires_grad=any(s["match_requires_grad"] for s in seen),
            match_float_requires_grad=any(s["match_float_requires_grad"] for s in seen),
            match_float_grad_fns=sorted({s["match_float_grad_fn"] for s in seen}),
            queue_dtypes=sorted({q["dtype"] for s in seen for q in s["queues"]}),
            queue_requires_grad=any(q["requires_grad"] for s in seen for q in s["queues"]),
            queue_grad_fns=sorted({q["grad_fn"] for s in seen for q in s["queues"]}),
            det_instance_feature_requires_grad=any(s["det_feature_requires_grad"] for s in seen),
            autocast_enabled_in_call=sorted({s["autocast_enabled"] for s in seen}),
        ),
    )
    for k, v in summary.items():
        print(f"{k}: {v}")
    C.dump_json(dict(summary=summary, checks=rows, training_calls=seen),
                os.path.join(C.OUT_ROOT, "probe_queue_grad.json"))


if __name__ == "__main__":
    torch.multiprocessing.set_start_method("fork")
    main()
