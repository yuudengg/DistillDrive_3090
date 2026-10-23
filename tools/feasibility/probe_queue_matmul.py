"""Extra probe (no retraining, no source edit): InstanceQueue.prepare_motion builds a
(B, 900, 900, C) broadcast product to apply a one-hot tracking match. Replace it
in-process with the equivalent matmul and compare latency / peak memory, plus an
operator-level check: on every call both forms are computed on the same inputs and
their difference recorded. (Final outputs are not compared elementwise: tiny
summation-order differences reorder the score-sorted boxes, which makes such a
diff meaningless; a full val eval of this variant was not run.)

  PYTHONPATH=. python tools/feasibility/probe_queue_matmul.py --mode fp32
"""
import argparse
import os

import torch

import common as C
from infer_bench import bench


def prepare_motion_matmul(self, det_output, mask):
    """Same as InstanceQueue.prepare_motion, with the broadcast-sum as matmul."""
    instance_feature = det_output["instance_feature"]
    det_anchors = det_output["prediction"][-1]
    if self.period == None:  # noqa: E711  (mirrors the original)
        self.period = instance_feature.new_zeros(instance_feature.shape[:2]).long()
    else:
        match = det_output["instance_id"][..., None] == self.prev_instance_id[:, None]
        if self.tracking_threshold > 0:
            match = match * (self.prev_confidence > self.tracking_threshold).unsqueeze(1)
        # autocast off: a one-hot gather must not round fp32 queue entries to fp16
        with torch.autocast(device_type="cuda", enabled=False):
            for i in range(len(self.instance_feature_queue)):
                f = self.instance_feature_queue[i]
                self.instance_feature_queue[i] = torch.matmul(match.to(f.dtype), f)
                a = self.anchor_queue[i]
                self.anchor_queue[i] = torch.matmul(match.to(a.dtype), a)
        self.period = (match * self.period[:, None]).sum(dim=2)
    self.instance_feature_queue.append(instance_feature.detach())
    self.anchor_queue.append(det_anchors.detach())
    self.period += 1
    if len(self.instance_feature_queue) > self.queue_length:
        self.instance_feature_queue.pop(0)
        self.anchor_queue.pop(0)
    self.period = torch.clip(self.period, 0, self.queue_length)


def make_checked(diffs):
    """Original prepare_motion, but also computes the matmul form and records |diff|."""

    def checked(self, det_output, mask):
        if self.period is not None:
            match = det_output["instance_id"][..., None] == self.prev_instance_id[:, None]
            if self.tracking_threshold > 0:
                match = match * (self.prev_confidence > self.tracking_threshold).unsqueeze(1)
            for q in list(self.instance_feature_queue) + list(self.anchor_queue):
                ref = (match[..., None] * q[:, None]).sum(dim=2)
                with torch.autocast(device_type="cuda", enabled=False):
                    new = torch.matmul(match.to(q.dtype), q)
                diffs.append(dict(dtype=str(q.dtype), max_abs=float((ref - new).abs().max()),
                                  ref_absmax=float(ref.abs().max()),
                                  max_matches_per_row=int(match.sum(-1).max())))
        return ORIGINAL(self, det_output, mask)

    return checked


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=C.PRECISION_MODES, default="fp32")
    ap.add_argument("--warmup", type=int, default=10)
    ap.add_argument("--iters", type=int, default=100)
    args = ap.parse_args()

    from projects.mmdet3d_plugin.models.motion.instance_queue import InstanceQueue

    cfg = C.load_cfg()
    _, loader = C.build_test_loader(cfg)
    samples = C.prefetch(loader, args.warmup + args.iters)
    global ORIGINAL
    ORIGINAL = InstanceQueue.prepare_motion
    report = dict(mode=args.mode)
    for variant in ["original", "matmul"]:
        InstanceQueue.prepare_motion = ORIGINAL if variant == "original" else prepare_motion_matmul
        model, _ = C.build_model(cfg)
        C.apply_precision(model, args.mode)
        model.cuda().eval()
        report[variant] = bench(model, samples, args.warmup)
        print(variant, report[variant])
        del model
        torch.cuda.empty_cache()

    diffs = []
    InstanceQueue.prepare_motion = make_checked(diffs)
    model, _ = C.build_model(cfg)
    C.apply_precision(model, args.mode)
    model.cuda().eval()
    with torch.no_grad():
        for d in samples:
            model(return_loss=False, rescale=True, **C.to_gpu(d))
    InstanceQueue.prepare_motion = ORIGINAL
    report["operator_check"] = dict(
        n_calls=len(diffs),
        dtypes=sorted({d["dtype"] for d in diffs}),
        max_abs_diff=max(d["max_abs"] for d in diffs),
        ref_absmax=max(d["ref_absmax"] for d in diffs),
        max_matches_per_row=max(d["max_matches_per_row"] for d in diffs),
    )
    print(report["operator_check"])
    C.dump_json(report, os.path.join(C.OUT_ROOT, f"probe_queue_matmul_{args.mode}.json"))


if __name__ == "__main__":
    torch.multiprocessing.set_start_method("fork")
    main()
