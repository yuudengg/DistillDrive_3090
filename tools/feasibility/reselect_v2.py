"""Step 6, plan 2 (v2): fewer tuned values to reduce over-fitting on the few collision cases.
CPU only, from dump_planning.py + risk_eval.py outputs. No training; the model is unchanged.

Rule (only model outputs are used to choose):
  normally        keep the model's choice (rescore, then highest score)
  trigger         pred_min_dist of the model's choice < tau
  when triggered  among the 6 commanded candidates whose pred_max_accel <= chosen's + ACC_X,
                  take the lexicographic best: lowest pred_col_prob, then largest pred_min_dist.
                  The model's choice is kept unless another candidate is strictly better.
Fixed in advance (before any result): ACC_X = 1.0 m/s^2, TAU_FIXED = 1.0 m (= the near-miss
distance), the tau grid, the 2-fold scene split of reselect.py (seed 0) and the tuning objective.
Two versions:
  tuned   one tau shared by all checkpoints: on the tuning fold pick the tau minimising the mean
          obj_box_col of the checkpoints subject to mean L2 <= mean baseline L2 + 0.02 m (ties:
          lower L2, then fewer changes); apply it to the other fold; merge the two folds.
  fixed   tau = TAU_FIXED, no tuning.
Significance: exact McNemar test per checkpoint on per-sample collision (any step, obj_box_col
rule) and near-miss, baseline vs plan 2.
Timing: a vectorised re-implementation of the online part (uniform-grid interpolation instead of
np.interp loops), checked against risk_eval.py's values, timed per frame.

  PYTHONPATH=. python tools/feasibility/reselect_v2.py --run D_seed0=work_dirs/risk_eval/D_seed0 ... \
      --out work_dirs/risk_eval/reselect_v2
"""
import argparse
import json
import os
import pickle
import time

import mmcv
import numpy as np
from scipy.stats import binomtest

from reselect import evaluate, load_run

ACC_X = 1.0
TAU_FIXED = 1.0
TAU_GRID = [0.25, 0.5, 0.75, 1.0, 1.5, 2.0, 3.0, np.inf]
L2_SLACK = 0.02


def choose(A, m, tau):
    n = len(A["sel"])
    r = np.arange(n)
    sel = A["sel"]
    trig = (A["md"][r, sel] < tau) & m
    allowed = A["acc"] <= A["acc"][r, sel][:, None] + ACC_X             # the chosen one is always allowed
    colp = np.where(allowed, A["colp"], np.inf)
    md = np.where(allowed, np.minimum(A["md"], 1e6), -np.inf)
    cmin = colp.min(-1, keepdims=True)
    tie = np.isclose(colp, cmin, rtol=0, atol=1e-9)
    md_t = np.where(tie, md, -np.inf)
    best = md_t.argmax(-1)
    sel_is_best = tie[r, sel] & (md[r, sel] >= md_t[r, best])
    new = np.where(sel_is_best, sel, best)
    return np.where(trig, new, sel), trig


def mcnemar(base_flag, new_flag):
    b = int((base_flag & ~new_flag).sum())     # fixed by plan 2
    c = int((~base_flag & new_flag).sum())     # introduced by plan 2
    p = binomtest(b, b + c, 0.5).pvalue if b + c > 0 else 1.0
    return dict(removed=b, introduced=c, p_value=float(p))


def flags(A, ch):
    r = np.arange(len(ch))
    return A["colt"][r, ch].any(-1), A["near"][r, ch].astype(bool)


def tune_shared(runs, folds, f_tune):
    best = None
    for tau in TAU_GRID:
        cols, l2s, l2b, chg = [], [], [], []
        for n, A in runs.items():
            m = folds[n] == f_tune
            ch, trig = choose(A, m, tau)
            e = evaluate(A, ch, m, trig)
            cols.append(e["obj_box_col"]); l2s.append(e["L2"]); chg.append(e["changed_rate"])
            l2b.append(evaluate(A, A["sel"], m)["L2"])
        if np.mean(l2s) > np.mean(l2b) + L2_SLACK:
            continue
        key = (np.mean(cols), np.mean(l2s), np.mean(chg))
        if best is None or key < best[0]:
            best = (key, tau)
    return best[1]


def run_version(runs, folds, tau_for_fold):
    out = {}
    for n, A in runs.items():
        full = np.ones(len(A["sel"]), bool)
        ch = A["sel"].copy()
        trig = np.zeros(len(ch), bool)
        for f in (0, 1):
            m = folds[n] == f
            c, t = choose(A, m, tau_for_fold[f])
            ch[m], trig[m] = c[m], t[m]
        base, new = evaluate(A, A["sel"], full), evaluate(A, ch, full, trig)
        bc, bn = flags(A, A["sel"])
        nc, nn = flags(A, ch)
        out[n] = dict(baseline=base, plan2=new, mcnemar_col=mcnemar(bc, nc), mcnemar_near=mcnemar(bn, nn))
    return out


# ---------------------------------------------------------------- vectorised online computation
def _interp_uniform(traj, step, td):
    """traj [..., K, 2] sampled at 0, step, 2*step, ... ; linear interpolation to td (within range)."""
    x = td / step
    i0 = np.minimum(np.floor(x).astype(int), traj.shape[-2] - 2)
    w = (x - i0)[:, None]
    return traj[..., i0, :] * (1 - w) + traj[..., i0 + 1, :] * w


def _path_yaw(xy, yaw0, eps=0.05):
    d = np.diff(xy, axis=-2)
    moving = np.linalg.norm(d, axis=-1) > eps
    yaw = np.arctan2(d[..., 1], d[..., 0])
    T = xy.shape[-2]
    idx = np.where(moving, np.arange(1, T), 0)                  # last moving step index (1-based)
    idx = np.maximum.accumulate(idx, axis=-1)
    full = np.concatenate([np.broadcast_to(np.asarray(yaw0)[..., None], yaw.shape[:-1] + (1,)), yaw], -1)
    out = np.take_along_axis(full, np.concatenate([np.zeros(idx.shape[:-1] + (1,), int), idx], -1), -1)
    return out


PRUNE_M = 10.0


def online_scores(rec, rows, v0, td, agent_thresh=0.3, prune=False):
    from risk_eval import rect_dist
    from risk_metrics import EGO_L, EGO_W, DT_PLAN, ego_center, obb_overlap
    plan = rec["plan_traj"][rows].astype(np.float64)
    M = len(rows)
    P = np.concatenate([np.zeros((M, 1, 2)), plan], 1)
    ego = _interp_uniform(P, DT_PLAN, td)
    yaw = _path_yaw(ego, np.full(M, np.pi / 2))
    ec = ego_center(ego, yaw)
    v = np.diff(P, axis=1) / DT_PLAN
    a = np.concatenate([(v[:, :1] - np.array([0, v0])) / (DT_PLAN / 2), np.diff(v, axis=1) / DT_PLAN], 1)
    acc = np.linalg.norm(a, axis=-1).max(-1)
    ag = rec["agents"]
    k = ag["cls_scores"] >= agent_thresh
    if not k.any():
        return np.zeros(M), np.full(M, np.inf), acc
    box = ag["boxes"][k].astype(np.float64)
    tr = ag["trajs"][k].astype(np.float64)
    p = ag["trajs_score"][k].astype(np.float64)
    p /= np.maximum(p.sum(-1, keepdims=True), 1e-9)
    if prune:
        # exact pruning: an agent whose centre stays more than PRUNE_M + both circumradii from every
        # ego centre on every mode can neither overlap nor be the minimum distance unless that
        # minimum is >= PRUNE_M; in that case fall back to all agents.
        full0 = np.concatenate([np.broadcast_to(box[:, None, None, :2], (len(box), 6, 1, 2)), tr], 2)
        axy0 = _interp_uniform(full0, DT_PLAN, td)
        rad = 0.5 * np.hypot(box[:, 3], box[:, 4]) + 0.5 * np.hypot(EGO_W, EGO_L)
        d = np.linalg.norm(axy0[None] - ec[:, None, None], axis=-1).min(axis=(-1, -2)).min(0)   # [N]
        near = d - rad < PRUNE_M
        if near.any():
            sub = {kk: vv[k][near] for kk, vv in ag.items()}
            colp_p, md_p, _ = online_scores(dict(rec, agents=dict(sub, cls_scores=np.ones(int(near.sum())))),
                                            rows, v0, td, agent_thresh=0.0)
            if (md_p < PRUNE_M).all():
                return colp_p, md_p, acc
    N = len(box)
    full = np.concatenate([np.broadcast_to(box[:, None, None, :2], (N, 6, 1, 2)), tr], 2)
    axy = _interp_uniform(full, DT_PLAN, td)
    ayaw = _path_yaw(axy, np.broadcast_to(box[:, None, 6], (N, 6)))
    ov = obb_overlap(ec[:, None, None], EGO_W, EGO_L, yaw[:, None, None], axy[None],
                     box[None, :, 3, None, None], box[None, :, 4, None, None], ayaw[None])
    colp = (ov.any(-1) * p[None]).sum(-1).max(-1)
    top = p.argmax(-1)
    g = rect_dist(ec[:, None], EGO_W, EGO_L, yaw[:, None], axy[np.arange(N), top][None],
                  box[None, :, 3, None], box[None, :, 4, None], ayaw[np.arange(N), top][None])
    return colp, g.min(axis=(-1, -2)), acc


def timing(dump_dir, risk_dir, n_frames, trigger_rates, dt=0.1):
    recs = mmcv.load(os.path.join(dump_dir, "planning_dump.pkl"))["records"][:n_frames]
    risk = mmcv.load(os.path.join(risk_dir, "risk_eval.pkl"))["risk"][:n_frames]
    with open("data/infos/nuscenes_infos_val.pkl", "rb") as f:
        infos = {x["token"]: x for x in pickle.load(f)["infos"]}
    td = np.arange(0, 3 + 1e-9, dt)
    t_trig, t_six, err = [], [], []
    p_trig, p_six, p_err = [], [], []
    for rec, rk in zip(recs, risk):
        c, s = rec["cmd"], rec["sel_mode"]
        v0 = float(infos[rec["token"]]["ego_status"][6])
        rows = list(range(c * 6, c * 6 + 6))
        ref = (rk["pred_col_prob"][rows], np.minimum(rk["pred_min_dist"][rows], 1e6), rk["pred_max_accel"][rows])
        for pr, (tt_, ts_, er_) in ((False, (t_trig, t_six, err)), (True, (p_trig, p_six, p_err))):
            t0 = time.perf_counter()
            online_scores(rec, [c * 6 + s], v0, td, prune=pr)
            t1 = time.perf_counter()
            colp, md, acc = online_scores(rec, rows, v0, td, prune=pr)
            t2 = time.perf_counter()
            tt_.append(t1 - t0); ts_.append(t2 - t1)
            er_.append(max(float(np.abs(a - b).max()) for a, b in zip((colp, np.minimum(md, 1e6), acc), ref)))
    st = lambda x: dict(mean_ms=float(np.mean(x) * 1e3), p50_ms=float(np.median(x) * 1e3), p95_ms=float(np.quantile(x, .95) * 1e3))
    out = dict(n_frames=len(recs), threads=os.environ.get("OMP_NUM_THREADS", "default"))
    for name, (a, b, e) in {"vectorised": (t_trig, t_six, err), "vectorised_pruned": (p_trig, p_six, p_err)}.items():
        tt, ts = float(np.mean(a) * 1e3), float(np.mean(b) * 1e3)
        out[name] = dict(trigger=st(a), rescore6=st(b), max_abs_diff_vs_risk_eval=float(max(e)),
                         added_ms_per_frame={k: tt + r * ts for k, r in trigger_rates.items()})
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", action="append", required=True, help="NAME=work_dirs/risk_eval/NAME")
    ap.add_argument("--out", required=True)
    ap.add_argument("--time-frames", type=int, default=1000)
    args = ap.parse_args()
    paths = dict(x.split("=", 1) for x in args.run)
    runs = {n: load_run(p) for n, p in paths.items()}
    scenes = sorted(set(next(iter(runs.values()))["scene"]))
    perm = np.random.RandomState(0).permutation(len(scenes))        # same split as reselect.py
    fold_of = {scenes[j]: int(k < len(scenes) // 2) for k, j in enumerate(perm)}
    folds = {n: np.array([fold_of[s] for s in A["scene"]]) for n, A in runs.items()}

    tau_tuned = {f: tune_shared(runs, folds, 1 - f) for f in (0, 1)}  # fold f evaluated with tau from 1 - f
    res = dict(acc_x=ACC_X, tau_fixed=TAU_FIXED, tau_grid=[float(t) for t in TAU_GRID], l2_slack=L2_SLACK,
               tau_tuned={f"eval_fold{f}": float(t) for f, t in tau_tuned.items()},
               tuned=run_version(runs, folds, tau_tuned),
               fixed=run_version(runs, folds, {0: TAU_FIXED, 1: TAU_FIXED}))
    for ver in ["tuned", "fixed"]:
        for n, o in res[ver].items():
            b, p = o["baseline"], o["plan2"]
            print(f"{ver:5s} {n}: col {b['obj_box_col']*100:.3f}->{p['obj_box_col']*100:.3f}% L2 {b['L2']:.4f}->{p['L2']:.4f} "
                  f"near {b['near']*100:.2f}->{p['near']*100:.2f}% trig {p['trigger_rate']:.3f} chg {p['changed_rate']:.3f} "
                  f"mcnemar col {o['mcnemar_col']} near p={o['mcnemar_near']['p_value']:.3g}")
    print("tau tuned:", res["tau_tuned"])
    first = next(iter(paths))
    rates = {f"{v}_{n}": res[v][n]["plan2"]["trigger_rate"] for v in ["tuned", "fixed"] for n in runs}
    res["timing"] = timing(paths[first], paths[first], args.time_frames, rates)
    print({k: ({kk: vv for kk, vv in v.items() if kk != "added_ms_per_frame"} if isinstance(v, dict) else v)
           for k, v in res["timing"].items()})
    os.makedirs(args.out, exist_ok=True)
    with open(os.path.join(args.out, "reselect_v2.json"), "w") as f:
        json.dump(res, f, indent=1, default=float)


if __name__ == "__main__":
    main()
