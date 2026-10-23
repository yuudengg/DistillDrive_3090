"""Layer-count test diagnosis: is the truncated model identical to the full model's
intermediate layer, and where does it diverge?

Runs the 6-layer model and an N-layer model (tools/feasibility/configs) side by side
on the same consecutive val frames (repo mode, matmul queue) and compares, per frame,
the full model's layer-N det/map outputs (classification[N-1], prediction[N-1]) with
the truncated model's last outputs. At a scene start the instance bank has no usable
cache, so both must agree up to the DeformableAggregation atomicAdd noise; afterwards
the truncated model caches its layer-N output (instead of layer 6) for the next frame.

  PYTHONPATH=. python tools/feasibility/probe_layer_exit.py --layers 5 --frames 80
"""
import argparse
import os

import numpy as np
import torch

import common as C


def capture(model):
    store = {}
    for attr in ("det_head", "map_head"):
        h = getattr(model.head, attr)
        h.register_forward_hook(lambda m, i, o, a=attr: store.__setitem__(a, o))
    return store


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--layers", type=int, default=5)
    ap.add_argument("--frames", type=int, default=80)
    args = ap.parse_args()

    C.set_queue_variant("matmul")
    cfg6 = C.load_cfg()
    # --layers 6 compares the full model with a second full model: run-to-run noise floor
    cfgN = cfg6 if args.layers == 6 else \
        C.load_cfg(f"tools/feasibility/configs/stage2_distribution_layers{args.layers}.py")
    models, stores = {}, {}
    for name, cfg in (("full", cfg6), ("cut", cfgN)):
        m, _ = C.build_model(cfg)
        C.apply_precision(m, "repo")
        m.cuda().eval()
        models[name], stores[name] = m, capture(m)

    _, loader = C.build_test_loader(cfg6)
    k = args.layers - 1
    rows, prev_ts = [], None
    with torch.no_grad():
        for f, data in enumerate(loader):
            if f >= args.frames:
                break
            d = C.to_gpu(data)
            ts = float(d["timestamp"][0]) if torch.is_tensor(d["timestamp"]) else float(d["timestamp"])
            scene_start = prev_ts is None or abs(ts - prev_ts) > 2.0
            prev_ts = ts
            for name in ("full", "cut"):
                models[name](return_loss=False, rescale=True, **d)
            row = dict(frame=f, scene_start=scene_start)
            for attr, short in (("det_head", "det"), ("map_head", "map")):
                a, b = stores["full"][attr], stores["cut"][attr]
                assert len(b["classification"]) == args.layers
                row[f"{short}_cls"] = float((a["classification"][k] - b["classification"][-1]).abs().max())
                row[f"{short}_reg"] = float((a["prediction"][k] - b["prediction"][-1]).abs().max())
                # how far the cut model's cached output is from what the full model caches
                row[f"{short}_feat_vs_full_last"] = float(
                    (a["instance_feature"] - b["instance_feature"]).abs().mean())
            rows.append(row)
            print(f"frame {f:3d} {'START' if scene_start else '     '} "
                  f"det cls {row['det_cls']:.2e} reg {row['det_reg']:.2e} | "
                  f"map cls {row['map_cls']:.2e} reg {row['map_reg']:.2e} | "
                  f"cached feat |full L6 - cut L{args.layers}| mean det {row['det_feat_vs_full_last']:.3f} "
                  f"map {row['map_feat_vs_full_last']:.3f}", flush=True)

    starts = [r for r in rows if r["scene_start"]]
    rest = [r for r in rows if not r["scene_start"]]
    summ = {}
    for key in ("det_cls", "det_reg", "map_cls", "map_reg"):
        summ[key] = dict(scene_start_max=max(r[key] for r in starts),
                         other_frames_median=float(np.median([r[key] for r in rest])),
                         other_frames_max=max(r[key] for r in rest))
    print(summ)
    C.dump_json(dict(layers=args.layers, frames=len(rows), summary=summ, per_frame=rows),
                os.path.join(C.OUT_ROOT, f"probe_layer_exit_L{args.layers}.json"))


if __name__ == "__main__":
    torch.multiprocessing.set_start_method("fork")
    main()
