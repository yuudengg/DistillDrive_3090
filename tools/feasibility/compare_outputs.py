"""Exp A: per-sample comparison of two saved output pickles (queue_matmul_eval.py --out).

For every output key: max |a-b| over all samples (float) or number of differing
elements (int), how many samples differ at all, and the first differing sample.
Detection/map outputs are sorted by score, so a tiny score change can reorder
them; for those the score vectors are also compared after sorting (order-free).

  PYTHONPATH=. python tools/feasibility/compare_outputs.py A.pkl B.pkl --name orig_vs_matmul
"""
import argparse
import os

import mmcv
import numpy as np
import torch

import common as C

ORDER_FREE = ["scores_3d", "cls_scores", "scores"]


def as_array(v):
    if torch.is_tensor(v):
        return v.detach().cpu().numpy()
    if isinstance(v, (list, tuple)) and v and isinstance(v[0], np.ndarray):
        return [np.asarray(x) for x in v]
    return np.asarray(v)


def diff(a, b):
    """(max_abs_diff or n_differing for ints, shapes_equal)"""
    if isinstance(a, list):
        if len(a) != len(b):
            return float("inf"), False
        ds = [diff(x, y) for x, y in zip(a, b)]
        return max([d for d, _ in ds] or [0.0]), all(ok for _, ok in ds)
    if a.shape != b.shape:
        return float("inf"), False
    if a.size == 0:
        return 0.0, True
    if np.issubdtype(a.dtype, np.floating):
        return float(np.abs(a.astype(np.float64) - b.astype(np.float64)).max()), True
    return float((a != b).sum()), True


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("a")
    ap.add_argument("b")
    ap.add_argument("--name", required=True)
    args = ap.parse_args()

    ra, rb = mmcv.load(args.a), mmcv.load(args.b)
    assert len(ra) == len(rb), (len(ra), len(rb))
    stats = {}
    for i, (sa, sb) in enumerate(zip(ra, rb)):
        sa, sb = sa["img_bbox"], sb["img_bbox"]
        assert sa.keys() == sb.keys()
        for k in sa:
            x, y = as_array(sa[k]), as_array(sb[k])
            d, same_shape = diff(x, y)
            s = stats.setdefault(k, dict(max=0.0, n_samples_diff=0, first_diff=None,
                                         shape_mismatch=0, per_sample=[]))
            s["max"] = max(s["max"], d)
            s["per_sample"].append(d)
            s["shape_mismatch"] += int(not same_shape)
            if d > 0:
                s["n_samples_diff"] += 1
                if s["first_diff"] is None:
                    s["first_diff"] = i
            if k in ORDER_FREE and same_shape and not isinstance(x, list):
                ds, _ = diff(np.sort(x), np.sort(y))
                o = stats.setdefault(k + "_sorted", dict(max=0.0, n_samples_diff=0, first_diff=None,
                                                         shape_mismatch=0, per_sample=[]))
                o["max"] = max(o["max"], ds)
                o["per_sample"].append(ds)
                if ds > 0:
                    o["n_samples_diff"] += 1
                    if o["first_diff"] is None:
                        o["first_diff"] = i

    out = dict(a=args.a, b=args.b, n_samples=len(ra), keys={})
    for k, s in stats.items():
        ps = np.array(s.pop("per_sample"))
        out["keys"][k] = dict(s, median_per_sample=float(np.median(ps)),
                              p99_per_sample=float(np.percentile(ps, 99)))
        print(f"{k:<22} max {s['max']:<12.6g} samples_diff {s['n_samples_diff']:5d}/{len(ra)} "
              f"first {s['first_diff']} median {np.median(ps):.3g}")
    C.dump_json(out, os.path.join(C.OUT_ROOT, "queue_eval", f"compare_{args.name}.json"))


if __name__ == "__main__":
    main()
