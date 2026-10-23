"""Exp 3 diagnosis: where full-head FP16 inference first produces inf/NaN.

Runs consecutive val frames (loader order) and records, per frame,
  * the first head submodule (execution order) whose output is non-finite,
  * non-finite outputs of anchor_projection (not an nn.Module, so wrapped
    directly) together with its time interval and max |velocity|.

  PYTHONPATH=. python tools/feasibility/probe_fp16_nan.py --mode full_amp_raw --frames 440
"""
import argparse
import os

import torch

import common as C


def flat_tensors(x):
    if torch.is_tensor(x):
        yield x
    elif isinstance(x, dict):
        for v in x.values():
            yield from flat_tensors(v)
    elif isinstance(x, (list, tuple)):
        for v in x:
            yield from flat_tensors(v)


def non_finite(x):
    return any(t.is_floating_point() and not torch.isfinite(t).all() for t in flat_tensors(x))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=C.PRECISION_MODES, default="full_amp_raw")
    ap.add_argument("--frames", type=int, default=440)
    args = ap.parse_args()

    cfg = C.load_cfg()
    model, _ = C.build_model(cfg)
    C.apply_precision(model, args.mode)
    model.cuda().eval()

    state = dict(frame=-1, first=None)
    proj_events = []

    def hook(mod, inp, out, name):
        if state["first"] is None and non_finite(out):
            origin = "input already non-finite" if non_finite(inp) else "created here"
            state["first"] = (name, type(mod).__name__, origin)

    for name, m in model.head.named_modules():
        if name:
            m.register_forward_hook(lambda mod, i, o, n=name: hook(mod, i, o, n))

    for hname in ("det_head", "map_head"):
        handler = getattr(model.head, hname).instance_bank.anchor_handler
        proj = handler.anchor_projection

        def spy(anchor, T_list, *a, _p=proj, _h=hname, **k):
            out = _p(anchor, T_list, *a, **k)
            if non_finite(out) and not non_finite(anchor):
                ti = k.get("time_intervals")
                proj_events.append(dict(
                    frame=state["frame"], head=_h,
                    time_interval_s=float(ti[0].flatten()[0]) if ti is not None else None,
                    anchor_absmax=float(anchor.abs().max()),
                    vel_absmax=float(anchor[..., 8:].abs().max()) if _h == "det_head" else None,
                ))
            return out

        handler.anchor_projection = spy

    _, loader = C.build_test_loader(cfg)
    bad_frames = []
    with torch.no_grad():
        for f, data in enumerate(loader):
            if f >= args.frames:
                break
            state["frame"], state["first"] = f, None
            model(return_loss=False, rescale=True, **C.to_gpu(data))
            if state["first"] is not None:
                bad_frames.append((f, state["first"]))

    lines = [f"mode={args.mode} frames={args.frames}",
             f"frames with non-finite head outputs: {len(bad_frames)}",
             f"first: {bad_frames[:2]}",
             f"anchor_projection created non-finite: {proj_events[:4]}"]
    print("\n".join(lines))
    with open(os.path.join(C.OUT_ROOT, f"probe_fp16_nan_{args.mode}.txt"), "w") as fh:
        fh.write("\n".join(lines) + "\n")


if __name__ == "__main__":
    torch.multiprocessing.set_start_method("fork")
    main()
