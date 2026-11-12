"""Step 6 (risk score): training-free risk metrics for all 18 planning candidates, on CPU,
from the per-sample dump written by dump_planning.py.

Two families per candidate (index = cmd * 6 + mode, same order as the model's [18, 6, 2] output):

  pred_*  deployable: uses only the model's own outputs (detected agents + their predicted
          multi-modal trajectories, predicted map boundaries). This is what a risk score may use.
    pred_ttc        first time (s) the ego box overlaps an agent box, agents moving on their most
                    likely mode; trajectories linearly interpolated to --dt; inf = none within 3 s
    pred_col_prob   max over agents of sum_k p_k * [ego overlaps the agent on mode k within 3 s]
                    (p_k = trajs_score normalised over the 6 modes)
    pred_min_dist   min clearance (m) between ego and agent footprints over 0..3 s (most likely mode),
                    3-circle footprint approximation; <= 0 means overlap
    pred_cross_bnd  ego centre path (0..3 s) crosses a predicted road-boundary polyline
    pred_bnd_margin min clearance (m) between the ego footprint and predicted boundaries
  gt_*    validation only (uses ground truth; never part of the score)
    gt_col_t [6]    ego box overlaps a GT future box at each 0.5 s step: same boxes, ego size,
                    0.5 m offset and yaw rule as planning_eval.PlanningMetric
    gt_ttc, gt_min_dist   as above against GT future boxes, at the 0.5 s steps
    gt_offroad_t [6] any ego-box corner outside the nuScenes drivable_area layer (--gt-map)

Boxes are in the lidar frame (x right, y forward); [x, y, z, w, l, h, yaw] with w along the box
heading (projects/mmdet3d_plugin/datasets/utils.py box3d_to_corners).

  PYTHONPATH=. python tools/feasibility/risk_metrics.py --dump .../planning_dump.pkl --gt-map
"""
import argparse
import json
import os
import time
from collections import defaultdict

import mmcv
import numpy as np

EGO_W, EGO_L, EGO_OFFSET = 4.084, 1.85, 0.5   # PlanningMetric: [H, W] -> box w (along heading), l
DT_PLAN = 0.5
N_CIRCLE = 3


# ---------------------------------------------------------------- geometry
def eval_yaw(traj):
    """planning_eval.get_yaw, vectorised over leading dims. traj [..., 6, 2] -> [..., 6]."""
    z = np.zeros(traj.shape[:-2] + (1, 2), traj.dtype)
    cat = np.concatenate([z, traj], axis=-2)                     # [..., 7, 2]
    yaw = np.zeros(cat.shape[:-1])
    yaw[..., 1:-1] = np.arctan2(cat[..., 2:, 1] - cat[..., :-2, 1], cat[..., 2:, 0] - cat[..., :-2, 0])
    yaw[..., -1] = np.arctan2(cat[..., -1, 1] - cat[..., -2, 1], cat[..., -1, 0] - cat[..., -2, 0])
    yaw = yaw[..., 1:]
    static = np.linalg.norm(traj[..., -1, :] - traj[..., 0, :], axis=-1) < 0.5
    return np.where(static[..., None], np.pi / 2, yaw)


def path_yaw(xy, yaw0, eps=0.05):
    """Heading along a dense path [..., T, 2]; keeps the previous heading while (almost) static."""
    d = np.diff(xy, axis=-2)
    step = np.linalg.norm(d, axis=-1)
    yaw = np.arctan2(d[..., 1], d[..., 0])
    out = np.empty(xy.shape[:-1])
    out[..., 0] = yaw0
    for t in range(1, xy.shape[-2]):
        out[..., t] = np.where(step[..., t - 1] > eps, yaw[..., t - 1], out[..., t - 1])
    return out


def ego_center(xy, yaw):
    """PlanningMetric shifts the ego box 0.5 m forward along its yaw."""
    return xy + EGO_OFFSET * np.stack([np.cos(yaw), np.sin(yaw)], -1)


def obb_overlap(c1, w1, l1, y1, c2, w2, l2, y2):
    """Separating-axis test for 2-D oriented boxes, broadcasting. c [..., 2]; w along heading."""
    def axes(y):
        return np.stack([np.cos(y), np.sin(y)], -1), np.stack([-np.sin(y), np.cos(y)], -1)
    a1, b1 = axes(y1)
    a2, b2 = axes(y2)
    d = c2 - c1
    sep = np.zeros(np.broadcast(d[..., 0], w1, w2).shape, bool)
    for ax in (a1, b1, a2, b2):
        dist = np.abs((d * ax).sum(-1))
        r1 = 0.5 * w1 * np.abs((a1 * ax).sum(-1)) + 0.5 * l1 * np.abs((b1 * ax).sum(-1))
        r2 = 0.5 * w2 * np.abs((a2 * ax).sum(-1)) + 0.5 * l2 * np.abs((b2 * ax).sum(-1))
        sep |= dist > r1 + r2
    return ~sep


def circles(c, w, l, yaw):
    """N_CIRCLE circles covering a w x l box (w along heading). -> centres [..., n, 2], radius [...]."""
    w, l = np.asarray(w, float), np.asarray(l, float)
    seg = w / N_CIRCLE
    r = np.sqrt((seg / 2) ** 2 + (l / 2) ** 2)
    k = (np.arange(N_CIRCLE) - (N_CIRCLE - 1) / 2)                # -1, 0, 1
    h = np.stack([np.cos(yaw), np.sin(yaw)], -1)
    cc = c[..., None, :] + (seg[..., None] * k)[..., None] * h[..., None, :]
    return cc, r


def clearance(c1, w1, l1, y1, c2, w2, l2, y2):
    p1, r1 = circles(c1, w1, l1, y1)
    p2, r2 = circles(c2, w2, l2, y2)
    d = np.linalg.norm(p1[..., :, None, :] - p2[..., None, :, :], axis=-1).min(axis=(-1, -2))
    return d - r1 - r2


def seg_dist(p, a, b):
    """Distance from points p [..., 2] to segments a, b [S, 2] -> [..., S]."""
    ab = b - a
    t = np.clip(((p[..., None, :] - a) * ab).sum(-1) / np.maximum((ab * ab).sum(-1), 1e-9), 0, 1)
    proj = a + t[..., None] * ab
    return np.linalg.norm(p[..., None, :] - proj, axis=-1)


def segs_intersect(p1, p2, q1, q2):
    """Proper intersection of segments p1p2 [..., 2] with q1q2 [S, 2] -> [..., S]."""
    def cross(o, a, b):
        return (a[..., 0] - o[..., 0]) * (b[..., 1] - o[..., 1]) - (a[..., 1] - o[..., 1]) * (b[..., 0] - o[..., 0])
    P1, P2 = p1[..., None, :], p2[..., None, :]
    d1, d2 = cross(q1, q2, P1), cross(q1, q2, P2)
    d3, d4 = cross(P1, P2, q1), cross(P1, P2, q2)
    return (d1 * d2 < 0) & (d3 * d4 < 0)


def interp(traj, t_src, t_dst):
    """Linear interpolation along the time axis (-2). traj [..., T, 2]."""
    flat = traj.reshape(-1, traj.shape[-2], 2)
    out = np.stack([np.stack([np.interp(t_dst, t_src, f[:, j]) for j in range(2)], -1) for f in flat])
    return out.reshape(traj.shape[:-2] + (len(t_dst), 2))


# ---------------------------------------------------------------- per sample
def pred_metrics(rec, args, t_dense):
    M = 18
    plan = rec["plan_traj"].astype(np.float64)                   # [18, 6, 2]
    t_plan = np.arange(0, 6 + 1) * DT_PLAN
    ego = interp(np.concatenate([np.zeros((M, 1, 2)), plan], 1), t_plan, t_dense)   # [18, T, 2]
    ego_yaw = path_yaw(ego, np.pi / 2)
    ego_c = ego_center(ego, ego_yaw)

    out = dict(pred_ttc=np.full(M, np.inf), pred_col_prob=np.zeros(M), pred_min_dist=np.full(M, np.inf),
               pred_cross_bnd=np.zeros(M, bool), pred_bnd_margin=np.full(M, np.inf))
    ag = rec["agents"]
    keep = ag["cls_scores"] >= args.agent_thresh
    if keep.any():
        box = ag["boxes"][keep].astype(np.float64)               # [N, 9]
        trajs = ag["trajs"][keep].astype(np.float64)             # [N, 6, 12, 2]
        p = ag["trajs_score"][keep].astype(np.float64)
        p = p / np.maximum(p.sum(-1, keepdims=True), 1e-9)       # [N, 6]
        N = box.shape[0]
        t_ag = np.arange(0, trajs.shape[2] + 1) * DT_PLAN
        cur = np.broadcast_to(box[:, None, None, :2], (N, 6, 1, 2))
        ag_xy = interp(np.concatenate([cur, trajs], 2), t_ag, t_dense)            # [N, 6, T, 2]
        ag_yaw = path_yaw(ag_xy, np.broadcast_to(box[:, None, 6], (N, 6)))
        # broadcast: [18, 1, 1, T] x [1, N, 6, T]
        E = lambda a: a[:, None, None]
        A = lambda a: a[None]
        bw, bl = box[:, 3][:, None, None], box[:, 4][:, None, None]
        ov = obb_overlap(E(ego_c), EGO_W, EGO_L, E(ego_yaw), A(ag_xy), A(bw), A(bl), A(ag_yaw))  # [18,N,6,T]
        hit_mode = ov.any(-1)                                    # [18, N, 6]
        out["pred_col_prob"] = (hit_mode * p[None]).sum(-1).max(-1)
        top = p.argmax(-1)                                       # [N]
        ov_top = ov[:, np.arange(N), top]                        # [18, N, T]
        first = np.where(ov_top.any(-1), ov_top.argmax(-1), -1)  # [18, N]
        tt = np.where(first >= 0, t_dense[np.maximum(first, 0)], np.inf)
        out["pred_ttc"] = tt.min(-1)
        xy_top = ag_xy[np.arange(N), top]                        # [N, T, 2]
        yaw_top = ag_yaw[np.arange(N), top]
        cl = clearance(ego_c[:, None], EGO_W, EGO_L, ego_yaw[:, None],
                       xy_top[None], bw[:, :, 0][None], bl[:, :, 0][None], yaw_top[None])  # [18, N, T]
        out["pred_min_dist"] = cl.min(axis=(-1, -2))

    mp = rec["map"]
    if mp["vectors"] is not None:
        sel = (mp["labels"] == 2) & (mp["scores"] >= args.map_thresh)
        if sel.any():
            v = mp["vectors"][sel].astype(np.float64)            # [K, 20, 2]
            a, b = v[:, :-1].reshape(-1, 2), v[:, 1:].reshape(-1, 2)
            out["pred_cross_bnd"] = segs_intersect(ego[:, :-1], ego[:, 1:], a, b).any(axis=(-1, -2))
            cc, r = circles(ego_c, np.full(ego_yaw.shape, EGO_W), np.full(ego_yaw.shape, EGO_L), ego_yaw)
            d = seg_dist(cc, a, b).min(-1)                       # [18, T, n]
            out["pred_bnd_margin"] = (d - r[..., None]).min(axis=(-1, -2))
    return out


def gt_metrics(rec):
    M = 18
    plan = rec["plan_traj"].astype(np.float64)
    yaw = eval_yaw(plan)                                         # [18, 6]
    c = ego_center(plan, yaw)
    col_t = np.zeros((M, 6), bool)
    dist_t = np.full((M, 6), np.inf)
    for t, fb in enumerate(rec["gt_fut_boxes"][:6]):
        if len(fb) == 0:
            continue
        fb = fb.astype(np.float64)
        args_b = (fb[None, :, :2], fb[None, :, 3], fb[None, :, 4], fb[None, :, 6])
        col_t[:, t] = obb_overlap(c[:, t, None], EGO_W, EGO_L, yaw[:, t, None], *args_b).any(-1)
        dist_t[:, t] = clearance(c[:, t, None], EGO_W, EGO_L, yaw[:, t, None], *args_b).min(-1)
    first = np.where(col_t.any(-1), col_t.argmax(-1), -1)
    return dict(gt_col_t=col_t, gt_min_dist=dist_t.min(-1),
                gt_ttc=np.where(first >= 0, (first + 1) * DT_PLAN, np.inf), _yaw=yaw, _c=c)


class DrivableArea:
    """nuScenes drivable_area polygons, unioned per scene around the ego positions."""

    def __init__(self, dataroot):
        self.dataroot, self.maps, self.cache = dataroot, {}, {}

    def scene_geom(self, scene, location, centers):
        if scene not in self.cache:
            from nuscenes.map_expansion.map_api import NuScenesMap
            from shapely.ops import unary_union
            from shapely.prepared import prep
            if location not in self.maps:
                self.maps[location] = NuScenesMap(dataroot=self.dataroot, map_name=location)
            m = self.maps[location]
            lo, hi = centers.min(0) - 80, centers.max(0) + 80
            toks = m.get_records_in_patch((lo[0], lo[1], hi[0], hi[1]), ["drivable_area"], mode="intersect")
            polys = [m.extract_polygon(pt) for rt in toks["drivable_area"]
                     for pt in m.get("drivable_area", rt)["polygon_tokens"]]
            geom = unary_union(polys)
            self.cache[scene] = (geom, prep(geom))
        return self.cache[scene][0]


def box_corners(c, w, l, yaw):
    h = np.stack([np.cos(yaw), np.sin(yaw)], -1)
    n = np.stack([-np.sin(yaw), np.cos(yaw)], -1)
    s = [(1, 1), (1, -1), (-1, -1), (-1, 1)]
    return np.stack([c + a * 0.5 * w * h + b * 0.5 * l * n for a, b in s], -2)   # [..., 4, 2]


def offroad(rec, gtm, da, scene_centers):
    from shapely import vectorized
    corners = box_corners(gtm["_c"], EGO_W, EGO_L, gtm["_yaw"])  # [18, 6, 4, 2] lidar
    T = rec["lidar2global"]
    g = corners @ T[:2, :2].T + T[:2, 3]
    geom = da.scene_geom(rec["scene_token"], rec["map_location"], scene_centers[rec["scene_token"]])
    inside = vectorized.contains(geom, g[..., 0].ravel(), g[..., 1].ravel()).reshape(g.shape[:-1])
    return ~inside.all(-1)                                       # [18, 6] any corner outside


# ---------------------------------------------------------------- summary
def auroc(score, label):
    label = np.asarray(label, bool)
    if label.all() or (~label).all():
        return None
    order = np.argsort(score, kind="mergesort")
    ranks = np.empty(len(score))
    s = np.asarray(score)[order]
    i = 0
    while i < len(s):                                            # average ranks for ties
        j = i
        while j + 1 < len(s) and s[j + 1] == s[i]:
            j += 1
        ranks[order[i:j + 1]] = (i + j) / 2 + 1
        i = j + 1
    npos = label.sum()
    return float((ranks[label].sum() - npos * (npos + 1) / 2) / (npos * (len(s) - npos)))


def summarize(records, risk, has_map):
    v = [i for i, r in enumerate(records) if r["valid"]]
    s = {"n_samples": len(records), "n_valid": len(v)}
    sel = lambda i: records[i]["cmd"] * 6 + records[i]["sel_mode"]
    # 1. our GT collision check vs the official per-sample one, on the selected trajectory
    ours = np.array([risk[i]["gt_col_t"][sel(i)] & ~records[i]["gt_col"] for i in v])
    offi = np.array([records[i]["col"] for i in v])
    s["gt_col_check_agreement_steps"] = float((ours == offi).mean())
    s["gt_col_check_agreement_samples"] = float((ours.any(1) == offi.any(1)).mean())
    # 2. selected trajectory: how often each risk flag fires
    def frac(f):
        return float(np.mean([f(i) for i in v]))
    s["selected"] = dict(
        official_col_any=frac(lambda i: records[i]["col"].any()),
        pred_ttc_le_3s=frac(lambda i: np.isfinite(risk[i]["pred_ttc"][sel(i)])),
        pred_col_prob_gt_0=frac(lambda i: risk[i]["pred_col_prob"][sel(i)] > 0),
        pred_min_dist_lt_1m=frac(lambda i: risk[i]["pred_min_dist"][sel(i)] < 1.0),
        pred_cross_bnd=frac(lambda i: risk[i]["pred_cross_bnd"][sel(i)]),
        rescore_flagged=frac(lambda i: records[i]["rescore_col"][records[i]["sel_mode"]]),
    )
    if has_map:
        s["selected"]["gt_offroad_any"] = frac(lambda i: risk[i]["gt_offroad_t"][sel(i)].any())
        s["gt_offroad_any_gt_traj_note"] = "not computed for the GT trajectory"
    # 3. is a deployable metric informative about GT outcomes? candidate-level AUROC on the cmd row
    rows = defaultdict(list)
    for i in v:
        c = records[i]["cmd"]
        idx = slice(c * 6, c * 6 + 6)
        r = risk[i]
        rows["gt_col"] += list(r["gt_col_t"][idx].any(-1))
        if has_map:
            rows["gt_offroad"] += list(r["gt_offroad_t"][idx].any(-1))
        rows["neg_ttc"] += list(-np.minimum(r["pred_ttc"][idx], 99))
        rows["col_prob"] += list(r["pred_col_prob"][idx])
        rows["neg_min_dist"] += list(-np.clip(r["pred_min_dist"][idx], -50, 50))
        rows["cross_bnd"] += list(r["pred_cross_bnd"][idx].astype(float))
        rows["neg_bnd_margin"] += list(-np.clip(r["pred_bnd_margin"][idx], -50, 50))
        rows["neg_plan_logit"] += list(-records[i]["plan_cls_raw"][idx])
    s["auroc_cmd_row_candidates"] = {
        target: {k: auroc(np.array(rows[k]), rows[target])
                 for k in ["neg_ttc", "col_prob", "neg_min_dist", "cross_bnd", "neg_bnd_margin", "neg_plan_logit"]}
        for target in (["gt_col", "gt_offroad"] if has_map else ["gt_col"])}
    s["auroc_note"] = ("6 candidates of the commanded row per valid sample; higher score = predicted riskier; "
                       "neg_plan_logit = model's own ranking as a reference")
    return s


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dump", required=True)
    ap.add_argument("--out", default=None, help="default: <dump dir>/risk_metrics.pkl")
    ap.add_argument("--agent-thresh", type=float, default=0.3, help="min detection cls score")
    ap.add_argument("--map-thresh", type=float, default=0.3, help="min boundary score")
    ap.add_argument("--dt", type=float, default=0.1)
    ap.add_argument("--gt-map", action="store_true", help="also check the nuScenes drivable_area layer")
    ap.add_argument("--dataroot", default="data/nuscenes")
    ap.add_argument("--limit", type=int, default=None)
    args = ap.parse_args()

    d = mmcv.load(args.dump)
    records = d["records"][:args.limit]
    out = args.out or os.path.join(os.path.dirname(args.dump), "risk_metrics.pkl")
    t_dense = np.arange(0, 3 + 1e-9, args.dt)
    da = DrivableArea(args.dataroot) if args.gt_map else None
    centers = defaultdict(list)
    for r in records:
        centers[r["scene_token"]].append(r["lidar2global"][:2, 3])
    centers = {k: np.array(v) for k, v in centers.items()}

    t0 = time.time()
    risk = []
    prog = mmcv.ProgressBar(len(records))
    for rec in records:
        r = pred_metrics(rec, args, t_dense)
        g = gt_metrics(rec)
        if da is not None:
            r["gt_offroad_t"] = offroad(rec, g, da, centers)
        r.update({k: v for k, v in g.items() if not k.startswith("_")})
        risk.append(r)
        prog.update()
    summary = summarize(records, risk, da is not None)
    summary.update(dump=args.dump, params=vars(args), cpu_min=(time.time() - t0) / 60,
                   dump_meta={k: v for k, v in d["meta"].items() if k != "metric"},
                   dump_metric=d["meta"]["metric"])
    mmcv.dump(dict(params=vars(args), risk=risk), out)
    with open(os.path.splitext(out)[0] + "_summary.json", "w") as f:
        json.dump(summary, f, indent=2, default=float)
    print(json.dumps(summary, indent=2, default=float))


if __name__ == "__main__":
    main()
