"""Step 6 (risk score), part 2: per-candidate risk metrics for all 18 planning candidates, on CPU,
from a dump_planning.py dump. No training, no GPU.

Metrics are split into two families and never mixed:
  pred_*  deployable - only the model's own outputs (detected agents with their predicted modes,
          predicted map lines, planning logits, ego speed from ego_status, the driving command)
  gt_*    validation only - nuScenes ground truth (future GT boxes with class and instance id,
          lane centrelines and drivable area of the nuScenes map)

Per candidate (index = cmd * 6 + mode, same order as the model's [18, 6, 2] output); s = seconds.
  safety   pred/gt: ttc, min_dist, pet, drac, vru_ttc, vru_min_dist; pred: col_prob
           (drac only for agents ahead of the ego and inside its lateral corridor; overlap = drac_cap)
  comfort  pred: max_accel, max_decel, max_jerk, max_yaw_rate
  progress pred: path_len, fwd_disp
  command  pred: cmd_violation (final lateral offset outside the commanded region, the rule that
           defines the command in tools/data_converter/nuscenes_converter.py), cmd_margin
  uncert.  pred: row_entropy, row_margin, row_spread (of the candidate's command row),
           cand_nll, cand_dev (deviation from the row's top-scored candidate)
  map      pred: cross_bnd, bnd_margin, line_dist, lane_center_off, lane_head_err (predicted
           dividers/boundaries as a lane proxy - the model predicts no centrelines)
           gt: offroad, lane_dist, lane_head_err (nuScenes lane / lane_connector centrelines)
Labels (validation, fixed before looking at results):
  gt_col      ego box overlaps a GT future box at some 0.5 s step where the GT ego box does not
              (planning_eval.PlanningMetric obj_box_col rule)
  gt_near     gt_min_dist < 1.0 m (exact box-to-box clearance over the 6 steps; includes collisions)
  l2_ade      mean L2 to the GT trajectory over the 6 steps (top-5 % defined in risk_analysis.py)

Geometry: lidar frame (x right, y forward). Ego box as in PlanningMetric (4.084 x 1.85 m, shifted
0.5 m forward). Pred agent trajectories use the most likely mode, dense time grid --dt over 0..3 s;
GT metrics use the 0.5 s steps. inf = no event within 3 s.

  PYTHONPATH=. python tools/feasibility/risk_eval.py --dump work_dirs/risk_eval/D_seed0/planning_dump.pkl
"""
import argparse
import os
import pickle
import time
from multiprocessing import Pool

import mmcv
import numpy as np

from risk_metrics import (EGO_L, EGO_W, DT_PLAN, box_corners, ego_center, eval_yaw, interp,
                          obb_overlap, path_yaw, seg_dist, segs_intersect, clearance, circles)

VRU = {"pedestrian", "bicycle"}
VRU_IDX = {7, 8}                       # class_names order in the stage2 config
G = {}                                 # per-process globals (filled before fork)


# ---------------------------------------------------------------- geometry
def rect_dist(c1, w1, l1, y1, c2, w2, l2, y2):
    """Exact clearance between 2-D oriented boxes (0 when overlapping), broadcasting."""
    A = box_corners(c1, np.asarray(w1, float)[..., None], np.asarray(l1, float)[..., None], y1)   # [..., 4, 2]
    B = box_corners(c2, np.asarray(w2, float)[..., None], np.asarray(l2, float)[..., None], y2)
    shp = np.broadcast_shapes(A.shape, B.shape)
    A, B = np.broadcast_to(A, shp), np.broadcast_to(B, shp)

    def pts_to_edges(P, Q):
        a, b = Q, np.roll(Q, -1, axis=-2)                        # edges of Q
        p = P[..., :, None, :]
        ab = (b - a)[..., None, :, :]
        t = np.clip(((p - a[..., None, :, :]) * ab).sum(-1) / np.maximum((ab * ab).sum(-1), 1e-9), 0, 1)
        proj = a[..., None, :, :] + t[..., None] * ab
        return np.linalg.norm(p - proj, axis=-1).min(axis=(-1, -2))
    d = np.minimum(pts_to_edges(A, B), pts_to_edges(B, A))
    return np.where(obb_overlap(c1, w1, l1, y1, c2, w2, l2, y2), 0.0, d)


def in_path(ec, eyaw, axy, aw, al):
    """Agent centre ahead of the ego box and laterally inside its corridor (DRAC is defined for a
    vehicle on the ego's path, not one being passed). Broadcasting; returns bool."""
    h = np.stack([np.cos(eyaw), np.sin(eyaw)], -1)
    rel = axy - ec
    lon = (rel * h).sum(-1)
    lat = np.abs(h[..., 0] * rel[..., 1] - h[..., 1] * rel[..., 0])
    return (lon > 0) & (lat < EGO_L / 2 + np.maximum(aw, al) / 2)


def wrap(a):
    return (a + np.pi) % (2 * np.pi) - np.pi


def softmax(x, axis=-1):
    e = np.exp(x - x.max(axis, keepdims=True))
    return e / e.sum(axis, keepdims=True)


# ---------------------------------------------------------------- GT context from infos
def gt_steps(infos, i):
    """Replicates NuScenes3DDataset fut_boxes (use_valid_flag=False) for steps 0..6, adding class
    names and instance ids. Returns list of (boxes [n, 7] in the current lidar frame, names, ids)."""
    from projects.mmdet3d_plugin.datasets.nuscenes_3d_dataset import get_T_global
    cur = infos[i]
    T_cur = get_T_global(cur)
    out = []
    for k in range(0, 7):
        if i + k >= len(infos) or infos[i + k]["scene_token"] != cur["scene_token"]:
            break
        f = infos[i + k]
        m = f["num_lidar_pts"] > 0
        b = f["gt_boxes"][m].copy()
        T = np.linalg.inv(T_cur) @ get_T_global(f)
        b[:, :3] = b[:, :3] @ T[:3, :3].T + T[:3, 3]
        yv = np.stack([np.cos(b[:, 6]), np.sin(b[:, 6])], -1) @ T[:2, :2].T
        b[:, 6] = np.arctan2(yv[:, 1], yv[:, 0])
        out.append((b, f["gt_names"][m], np.asarray(f["instance_inds"])[m]))
    return out


def scene_geometry(dataroot, records, infos_by_token):
    """Per scene: nuScenes drivable_area union and lane/lane_connector centreline poses (global)."""
    from nuscenes.map_expansion.map_api import NuScenesMap
    from shapely.ops import unary_union
    maps, geo = {}, {}
    by_scene = {}
    for r in records:
        by_scene.setdefault(r["scene_token"], []).append(r)
    for sc, rs in by_scene.items():
        loc = rs[0]["map_location"]
        if loc not in maps:
            maps[loc] = NuScenesMap(dataroot=dataroot, map_name=loc)
        m = maps[loc]
        c = np.array([r["lidar2global"][:2, 3] for r in rs])
        lo, hi = c.min(0) - 80, c.max(0) + 80
        patch = (lo[0], lo[1], hi[0], hi[1])
        rec = m.get_records_in_patch(patch, ["drivable_area", "lane", "lane_connector"], mode="intersect")
        da = unary_union([m.extract_polygon(pt) for rt in rec["drivable_area"]
                          for pt in m.get("drivable_area", rt)["polygon_tokens"]])
        lanes = m.discretize_lanes(rec["lane"] + rec["lane_connector"], 0.5)
        poses = np.array([p for v in lanes.values() for p in v], dtype=np.float64).reshape(-1, 3)
        geo[sc] = (da, poses)
    return geo


# ---------------------------------------------------------------- per sample
def pred_family(rec, v0, args, t_dense):
    M, T = 18, len(t_dense)
    plan = rec["plan_traj"].astype(np.float64)
    P = np.concatenate([np.zeros((M, 1, 2)), plan], 1)                       # [18, 7, 2]
    ego = interp(P, np.arange(7) * DT_PLAN, t_dense)                         # [18, T, 2]
    eyaw = path_yaw(ego, np.pi / 2)
    ec = ego_center(ego, eyaw)
    o = {}
    INF = np.full(M, np.inf)

    # ---- agents (most likely mode)
    ag = rec["agents"]
    keep = ag["cls_scores"] >= args.agent_thresh
    for k in ["ttc", "min_dist", "pet", "vru_ttc", "vru_min_dist"]:
        o[f"pred_{k}"] = INF.copy()
    o["pred_drac"], o["pred_col_prob"] = np.zeros(M), np.zeros(M)
    if keep.any():
        box = ag["boxes"][keep].astype(np.float64)
        lab = ag["labels"][keep]
        trajs = ag["trajs"][keep].astype(np.float64)
        p = ag["trajs_score"][keep].astype(np.float64)
        p = p / np.maximum(p.sum(-1, keepdims=True), 1e-9)
        N = len(box)
        cur = np.broadcast_to(box[:, None, None, :2], (N, 6, 1, 2))
        axy = interp(np.concatenate([cur, trajs], 2), np.arange(trajs.shape[2] + 1) * DT_PLAN, t_dense)
        ayaw = path_yaw(axy, np.broadcast_to(box[:, None, 6], (N, 6)))
        bw, bl = box[:, 3], box[:, 4]
        ov = obb_overlap(ec[:, None, None], EGO_W, EGO_L, eyaw[:, None, None],
                         axy[None], bw[None, :, None, None], bl[None, :, None, None], ayaw[None])  # [18,N,6,T]
        o["pred_col_prob"] = (ov.any(-1) * p[None]).sum(-1).max(-1)
        top = p.argmax(-1)
        txy, tyaw = axy[np.arange(N), top], ayaw[np.arange(N), top]           # [N, T, 2], [N, T]
        ov1 = ov[:, np.arange(N), top]                                        # [18, N, T]
        first = np.where(ov1.any(-1), ov1.argmax(-1), -1)
        ttc_n = np.where(first >= 0, t_dense[np.maximum(first, 0)], np.inf)   # [18, N]
        gap = rect_dist(ec[:, None], EGO_W, EGO_L, eyaw[:, None],
                        txy[None], bw[None, :, None], bl[None, :, None], tyaw[None])   # [18, N, T]
        md_n = gap.min(-1)
        # PET: ego at t_e and agent at t_a occupy overlapping space
        ovp = obb_overlap(ec[:, None, :, None], EGO_W, EGO_L, eyaw[:, None, :, None],
                          txy[None, :, None], bw[None, :, None, None], bl[None, :, None, None],
                          tyaw[None, :, None])                                # [18, N, Te, Ta]
        dtm = np.abs(t_dense[:, None] - t_dense[None, :])
        pet_n = np.where(ovp, dtm, np.inf).min(axis=(-1, -2))                 # [18, N]
        # DRAC: closing speed^2 / (2 * gap)
        vc = -np.gradient(gap, args.dt, axis=-1)
        ahead = in_path(ec[:, None], eyaw[:, None], txy[None], bw[None, :, None], bl[None, :, None])
        drac = np.where((vc > 0) & (gap > 0) & ahead, vc ** 2 / (2 * np.maximum(gap, args.drac_min_gap)), 0.0)
        drac = np.where(gap <= 0, args.drac_cap, np.minimum(drac, args.drac_cap))
        o["pred_ttc"], o["pred_min_dist"] = ttc_n.min(-1), md_n.min(-1)
        o["pred_pet"], o["pred_drac"] = pet_n.min(-1), drac.max(axis=(-1, -2))
        vru = np.isin(lab, list(VRU_IDX))
        if vru.any():
            o["pred_vru_ttc"], o["pred_vru_min_dist"] = ttc_n[:, vru].min(-1), md_n[:, vru].min(-1)

    # ---- comfort / progress (0.5 s grid, v0 = current forward speed)
    vseg = np.diff(P, axis=1) / DT_PLAN                                       # [18, 6, 2] at .25, .75, ...
    v0v = np.array([0.0, v0])
    acc = np.concatenate([(vseg[:, :1] - v0v) / (DT_PLAN / 2), np.diff(vseg, axis=1) / DT_PLAN], 1)
    spd = np.linalg.norm(vseg, axis=-1)
    dspd = np.concatenate([(spd[:, :1] - abs(v0)) / (DT_PLAN / 2), np.diff(spd, axis=1) / DT_PLAN], 1)
    o["pred_max_accel"] = np.linalg.norm(acc, axis=-1).max(-1)
    o["pred_max_decel"] = np.maximum(-dspd, 0).max(-1)
    o["pred_max_jerk"] = np.linalg.norm(np.diff(acc, axis=1) / DT_PLAN, axis=-1).max(-1)
    hd = np.arctan2(vseg[..., 1], vseg[..., 0])
    hd_full = np.empty((M, 7))
    hd_full[:, 0] = np.pi / 2
    for k in range(6):
        hd_full[:, k + 1] = np.where(spd[:, k] > 1.0, hd[:, k], hd_full[:, k])
    o["pred_max_yaw_rate"] = np.abs(wrap(np.diff(hd_full, axis=1))).max(-1) / DT_PLAN
    o["pred_path_len"] = np.linalg.norm(np.diff(P, axis=1), axis=-1).sum(-1)
    o["pred_fwd_disp"] = P[:, -1, 1]

    # ---- command compliance (cmd is an input at inference)
    x = P[:, -1, 0]
    cmd = rec["cmd"]
    margin = {0: x - 2.0, 1: -2.0 - x, 2: 2.0 - np.abs(x)}[cmd]              # >= 0: inside the region
    o["pred_cmd_margin"] = margin
    o["pred_cmd_violation"] = (margin < 0).astype(float)

    # ---- uncertainty of the candidate's command row
    logit = rec["plan_cls_raw"].astype(np.float64).reshape(3, 6)
    pr = softmax(logit)
    ent = -(pr * np.log(np.maximum(pr, 1e-12))).sum(-1) / np.log(6)
    srt = np.sort(pr, -1)
    end = plan.reshape(3, 6, 6, 2)[:, :, -1]
    spread = np.linalg.norm(end[:, :, None] - end[:, None], axis=-1).sum((-1, -2)) / 30
    best = plan.reshape(3, 6, 6, 2)[np.arange(3), logit.argmax(-1)]           # [3, 6, 2]
    dev = np.linalg.norm(plan.reshape(3, 6, 6, 2) - best[:, None], axis=-1).mean(-1)
    o["pred_row_entropy"] = np.repeat(ent, 6)
    o["pred_row_margin_neg"] = np.repeat(-(srt[:, -1] - srt[:, -2]), 6)       # negated: higher = less sure
    o["pred_row_spread"] = np.repeat(spread, 6)
    o["pred_cand_nll"] = -np.log(np.maximum(pr, 1e-12)).reshape(18)
    o["pred_cand_dev"] = dev.reshape(18)

    # ---- predicted map lines
    mp = rec["map"]
    for k in ["line_dist", "bnd_margin"]:
        o[f"pred_{k}"] = INF.copy()
    o["pred_cross_bnd"] = np.zeros(M)
    o["pred_lane_center_off"], o["pred_lane_head_err"] = np.zeros(M), np.zeros(M)
    if mp["vectors"] is not None:
        sc_ok = mp["scores"] >= args.map_thresh
        v = mp["vectors"].astype(np.float64)
        bnd = sc_ok & (mp["labels"] == 2)
        lines = sc_ok & np.isin(mp["labels"], [1, 2])
        yaw6 = eval_yaw(plan)
        if bnd.any():
            a, b = v[bnd][:, :-1].reshape(-1, 2), v[bnd][:, 1:].reshape(-1, 2)
            o["pred_cross_bnd"] = segs_intersect(ego[:, :-1], ego[:, 1:], a, b).any(axis=(-1, -2)).astype(float)
            cc, r = circles(ec, np.full(eyaw.shape, EGO_W), np.full(eyaw.shape, EGO_L), eyaw)
            o["pred_bnd_margin"] = (seg_dist(cc, a, b).min(-1) - r[..., None]).min(axis=(-1, -2))
        if lines.any():
            a, b = v[lines][:, :-1].reshape(-1, 2), v[lines][:, 1:].reshape(-1, 2)
            d = seg_dist(plan, a, b)                                          # [18, 6, S]
            o["pred_line_dist"] = d.min(-1).min(-1)
            seg_dir = np.arctan2(b[:, 1] - a[:, 1], b[:, 0] - a[:, 0])
            near = d.argmin(-1)                                               # [18, 6]
            herr = np.abs(wrap(yaw6 - seg_dir[near]))
            herr = np.minimum(herr, np.pi - herr)                             # lines are undirected
            o["pred_lane_head_err"] = herr.max(-1)
            h = np.stack([np.cos(yaw6), np.sin(yaw6)], -1)
            rel = (a + b)[None, None] / 2 - plan[:, :, None]                  # [18, 6, S, 2]
            side = h[..., None, 0] * rel[..., 1] - h[..., None, 1] * rel[..., 0]   # >0 left
            dl = np.where(side > 0, d, np.inf).min(-1)
            dr = np.where(side <= 0, d, np.inf).min(-1)
            off = np.where(np.isfinite(dl) & np.isfinite(dr), np.abs(dl - dr) / 2, 0.0)
            o["pred_lane_center_off"] = off.max(-1)
    return o


def gt_family(rec, steps, scene_geo):
    from shapely import vectorized
    M = 18
    plan = rec["plan_traj"].astype(np.float64)
    yaw = eval_yaw(plan)
    c = ego_center(plan, yaw)                                                 # [18, 6, 2]
    o = {}
    col_t = np.zeros((M, 6), bool)
    dist_t = np.full((M, 6), np.inf)
    vdist_t = np.full((M, 6), np.inf)
    vcol_t = np.zeros((M, 6), bool)
    for t in range(1, len(steps)):
        b, names, _ = steps[t]
        if len(b) == 0:
            continue
        args_b = (b[None, :, :2], b[None, :, 3], b[None, :, 4], b[None, :, 6])
        ovb = obb_overlap(c[:, t - 1, None], EGO_W, EGO_L, yaw[:, t - 1, None], *args_b)   # [18, n]
        db = rect_dist(c[:, t - 1, None], EGO_W, EGO_L, yaw[:, t - 1, None], *args_b)
        col_t[:, t - 1], dist_t[:, t - 1] = ovb.any(-1), db.min(-1)
        vru = np.isin(names, list(VRU))
        if vru.any():
            vcol_t[:, t - 1], vdist_t[:, t - 1] = ovb[:, vru].any(-1), db[:, vru].min(-1)
    first = np.where(col_t.any(-1), col_t.argmax(-1), -1)
    vfirst = np.where(vcol_t.any(-1), vcol_t.argmax(-1), -1)
    o["gt_col_t"] = col_t
    o["gt_min_dist"] = dist_t.min(-1)
    o["gt_ttc"] = np.where(first >= 0, (first + 1) * DT_PLAN, np.inf)
    o["gt_vru_min_dist"] = vdist_t.min(-1)
    o["gt_vru_ttc"] = np.where(vfirst >= 0, (vfirst + 1) * DT_PLAN, np.inf)

    # PET and DRAC from instance tracks (steps 0..6, ego at the origin at t = 0)
    ego0 = np.zeros((M, 1, 2))
    ce = np.concatenate([ego_center(ego0, np.full((M, 1), np.pi / 2)), c], 1)  # [18, 7, 2]
    ye = np.concatenate([np.full((M, 1), np.pi / 2), yaw], 1)
    pet = np.full(M, np.inf)
    drac = np.zeros(M)
    tracks = {}
    for t, (b, _, ids) in enumerate(steps):
        for j, iid in enumerate(ids):
            tracks.setdefault(int(iid), {})[t] = b[j]
    for tr in tracks.values():
        ts = sorted(tr)
        bb = np.stack([tr[t] for t in ts])                                    # [k, 7]
        ta = np.array(ts)
        ovp = obb_overlap(ce[:, 1:, None], EGO_W, EGO_L, ye[:, 1:, None],
                          bb[None, None, :, :2], bb[None, None, :, 3], bb[None, None, :, 4], bb[None, None, :, 6])
        dtm = np.abs(np.arange(1, 7)[:, None] - ta[None, :]) * DT_PLAN
        pet = np.minimum(pet, np.where(ovp, dtm[None], np.inf).min(axis=(-1, -2)))
        g = rect_dist(ce[:, ta], EGO_W, EGO_L, ye[:, ta], bb[None, :, :2], bb[None, :, 3], bb[None, :, 4], bb[None, :, 6])
        if len(ta) >= 2:
            dt = np.diff(ta) * DT_PLAN
            vc = -np.diff(g, axis=-1) / dt
            gg = g[:, :-1]
            ahead = in_path(ce[:, ta[:-1]], ye[:, ta[:-1]], bb[None, :-1, :2], bb[None, :-1, 3], bb[None, :-1, 4])
            dr = np.where((vc > 0) & (gg > 0) & ahead, vc ** 2 / (2 * np.maximum(gg, G["args"].drac_min_gap)), 0.0)
            dr = np.where(gg <= 0, G["args"].drac_cap, np.minimum(dr, G["args"].drac_cap))
            drac = np.maximum(drac, dr.max(-1))
    o["gt_pet"], o["gt_drac"] = pet, drac

    # map: drivable area and lane centrelines (global frame)
    da, poses = scene_geo
    T = rec["lidar2global"]
    R = T[:2, :2]
    cor = box_corners(c, EGO_W, EGO_L, yaw) @ R.T + T[:2, 3]                  # [18, 6, 4, 2]
    inside = vectorized.contains(da, cor[..., 0].ravel(), cor[..., 1].ravel()).reshape(cor.shape[:-1])
    o["gt_offroad_t"] = ~inside.all(-1)
    if len(poses):
        from scipy.spatial import cKDTree
        tree = G["lane_tree"].get(rec["scene_token"])
        if tree is None:
            tree = G["lane_tree"][rec["scene_token"]] = cKDTree(poses[:, :2])
        pg = plan @ R.T + T[:2, 3]
        d, idx = tree.query(pg.reshape(-1, 2))
        gy = yaw + np.arctan2(R[1, 0], R[0, 0])
        herr = np.abs(wrap(gy.ravel() - poses[idx, 2]))
        o["gt_lane_dist"] = d.reshape(18, 6).max(-1)
        o["gt_lane_head_err"] = herr.reshape(18, 6).max(-1)
    else:
        o["gt_lane_dist"] = np.full(M, np.nan)
        o["gt_lane_head_err"] = np.full(M, np.nan)
    return o


def work(i):
    rec = G["records"][i]
    info = G["infos"][i]
    steps = gt_steps(G["infos"], i)
    o = pred_family(rec, float(info["ego_status"][6]), G["args"], G["t_dense"])
    o.update(gt_family(rec, steps, G["geo"][rec["scene_token"]]))
    # sanity: our GT boxes are the dump's (planning-eval) boxes
    fb = rec["gt_fut_boxes"]
    o["_gt_boxes_match"] = (len(fb) == len(steps) - 1) and all(
        len(a) == len(s[0]) and np.allclose(a, s[0][:, :7], atol=1e-3) for a, s in zip(fb, steps[1:]))
    if rec["valid"]:
        o["l2_ade"] = np.linalg.norm(rec["plan_traj"] - rec["gt_traj"][None], axis=-1).mean(-1)
        o["gt_col"] = (o["gt_col_t"] & ~rec["gt_col"][None]).any(-1)
        o["gt_near"] = o["gt_min_dist"] < G["args"].near_thresh
    return o


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dump", required=True)
    ap.add_argument("--out", default=None, help="default: <dump dir>/risk_eval.pkl")
    ap.add_argument("--infos", default="data/infos/nuscenes_infos_val.pkl")
    ap.add_argument("--dataroot", default="data/nuscenes")
    ap.add_argument("--agent-thresh", type=float, default=0.3)
    ap.add_argument("--map-thresh", type=float, default=0.3)
    ap.add_argument("--dt", type=float, default=0.1)
    ap.add_argument("--near-thresh", type=float, default=1.0, help="near-miss: GT min distance < this (m)")
    ap.add_argument("--drac-min-gap", type=float, default=0.1)
    ap.add_argument("--drac-cap", type=float, default=20.0)
    ap.add_argument("--workers", type=int, default=16)
    ap.add_argument("--limit", type=int, default=None)
    args = ap.parse_args()

    t0 = time.time()
    d = mmcv.load(args.dump)
    records = d["records"][:args.limit]
    with open(args.infos, "rb") as f:
        infos = sorted(pickle.load(f)["infos"], key=lambda e: e["timestamp"])   # dataset order
    assert [x["token"] for x in infos[:len(records)]] == [r["token"] for r in records]
    G.update(records=records, infos=infos, args=args, t_dense=np.arange(0, 3 + 1e-9, args.dt),
             lane_tree={})
    G["geo"] = scene_geometry(args.dataroot, records, None)
    print(f"geometry ready ({time.time() - t0:.0f}s)")
    with Pool(args.workers) as pool:
        res = list(mmcv.track_iter_progress((pool.imap(work, range(len(records)), chunksize=8), len(records))))
    out = args.out or os.path.join(os.path.dirname(args.dump), "risk_eval.pkl")
    meta = dict(dump=args.dump, params=vars(args), dump_meta=d["meta"], cpu_min=(time.time() - t0) / 60,
                gt_boxes_match=float(np.mean([r.pop("_gt_boxes_match") for r in res])))
    sel = [r["cmd"] * 6 + r["sel_mode"] for r in records]
    mmcv.dump(dict(meta=meta, risk=res, sel=sel, valid=[r["valid"] for r in records],
                   scene=[r["scene_token"] for r in records], token=[r["token"] for r in records],
                   official_col=[r["col"].any() if r["valid"] else False for r in records]), out)
    print({k: v for k, v in meta.items() if k != "dump_meta"})


if __name__ == "__main__":
    main()
