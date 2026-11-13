"""Step 6, plan 2: risk-triggered re-selection among the 6 candidates of the current command.
CPU only, from dump_planning.py + risk_eval.py outputs. No training; the model is unchanged.

Rule (deployable: only model outputs are used to choose):
  normally         keep the model's choice (rescore, then highest score)
  trigger          pred_min_dist of the model's choice < tau
  when triggered   pick argmax over the 6 commanded candidates of
                       -pred_col_prob + w_d * min(pred_min_dist, D_CAP) - w_a * pred_max_accel
                   keeping the model's choice on ties (so it changes only when strictly better)
Tuning (fixed before looking at the result): the 150 val scenes are split 2-fold by a fixed random
permutation of scene tokens (seed 0). On the tuning fold pick (tau, w_d, w_a) from the grid that
minimises obj_box_col subject to L2 <= baseline L2 + L2_SLACK on that fold (ties: lower L2, then
fewer changes); apply it to the other fold. Each fold is evaluated with parameters tuned on the other
one and the two halves are merged into full-val numbers. GT is used only for evaluation/tuning.

obj_box_col and L2 follow planning_eval exactly (per-step mean over valid samples, cumulative mean
over steps, average of 1 / 2 / 3 s).

  PYTHONPATH=. python tools/feasibility/reselect.py --run D_seed0=work_dirs/risk_eval/D_seed0 ... \
      --out work_dirs/risk_eval/reselect
"""
import argparse
import itertools
import json
import os
import pickle
import time

import mmcv
import numpy as np
from sklearn.metrics import average_precision_score, roc_auc_score

D_CAP = 5.0
L2_SLACK = 0.02
GRID = dict(tau=[0.25, 0.5, 0.75, 1.0, 1.5, 2.0, 3.0, np.inf],
            w_d=[0.0, 0.01, 0.03, 0.1, 0.3],
            w_a=[0.0, 0.01, 0.03, 0.1, 0.3])


def load_run(d):
    dump = mmcv.load(os.path.join(d, "planning_dump.pkl"))["records"]
    risk = mmcv.load(os.path.join(d, "risk_eval.pkl"))["risk"]
    vi = [i for i, r in enumerate(dump) if r["valid"]]
    A = {k: [] for k in ["colp", "md", "acc", "jerk", "plen", "near", "colt", "l2t", "sel", "scene", "idx"]}
    for i in vi:
        r, k = dump[i], risk[i]
        row = slice(r["cmd"] * 6, r["cmd"] * 6 + 6)
        A["colp"].append(k["pred_col_prob"][row]); A["md"].append(k["pred_min_dist"][row])
        A["acc"].append(k["pred_max_accel"][row]); A["jerk"].append(k["pred_max_jerk"][row])
        A["plen"].append(k["pred_path_len"][row]); A["near"].append(k["gt_near"][row])
        A["colt"].append(k["gt_col_t"][row] & ~r["gt_col"][None])                 # [6, 6] obj_box_col rule
        A["l2t"].append(np.linalg.norm(r["plan_traj"][row] - r["gt_traj"][None], axis=-1))  # [6, 6]
        A["sel"].append(r["sel_mode"]); A["scene"].append(r["scene_token"]); A["idx"].append(i)
    return {k: np.array(v) for k, v in A.items()}


def official(per_sample_t):
    per_t = per_sample_t.astype(np.float64).mean(0)
    cum = np.cumsum(per_t) / np.arange(1, len(per_t) + 1)
    return float(cum[[1, 3, 5]].mean())


def choose(A, m, tau, w_d, w_a):
    n = len(A["sel"])
    sel = A["sel"]
    trig = A["md"][np.arange(n), sel] < tau
    s = -A["colp"] + w_d * np.minimum(A["md"], D_CAP) - w_a * A["acc"]
    best = s.argmax(-1)
    keep = s[np.arange(n), sel] >= s[np.arange(n), best] - 1e-12
    new = np.where(keep, sel, best)
    ch = np.where(trig & m, new, sel)
    return ch, trig


def evaluate(A, ch, m, trig=None):
    r = np.arange(len(ch))[m]
    c = ch[m]
    o = dict(n=int(m.sum()), obj_box_col=official(A["colt"][r, c]), L2=official(A["l2t"][r, c]),
             near=float(A["near"][r, c].mean()), path_len=float(A["plen"][r, c].mean()),
             max_accel=float(A["acc"][r, c].mean()), max_jerk=float(A["jerk"][r, c].mean()),
             col_samples=int(A["colt"][r, c].any(-1).sum()))
    if trig is not None:
        o["trigger_rate"] = float(trig[m].mean())
        o["changed_rate"] = float((c != A["sel"][m]).mean())
    return o


def tune(A, m):
    base = evaluate(A, A["sel"], m)
    best = None
    for tau, w_d, w_a in itertools.product(GRID["tau"], GRID["w_d"], GRID["w_a"]):
        ch, trig = choose(A, m, tau, w_d, w_a)
        e = evaluate(A, ch, m, trig)
        if e["L2"] > base["L2"] + L2_SLACK:
            continue
        key = (e["obj_box_col"], e["L2"], e["changed_rate"])
        if best is None or key < best[0]:
            best = (key, dict(tau=tau, w_d=w_d, w_a=w_a))
    return best[1]


def auroc_rows(runs):
    """Current-command 6 candidates: AUROC / PR-AUC of the three re-selection metrics."""
    out = {}
    feats = {"pred_col_prob": ("colp", +1), "pred_min_dist": ("md", -1), "pred_max_accel": ("acc", +1)}
    pooled = {k: [] for k in list(feats) + ["col", "near", "l2top5"]}
    per = {}
    for name, A in runs.items():
        col = A["colt"].any(-1).ravel()
        near = A["near"].ravel()
        l2 = A["l2t"].mean(-1).ravel()
        lab = dict(col=col, near=near, l2top5=l2 >= np.quantile(l2, 0.95))
        per[name] = {}
        for f, (k, sgn) in feats.items():
            x = np.nan_to_num(sgn * A[k].ravel(), posinf=1e6, neginf=-1e6)
            pooled[f].append(x)
            per[name][f] = {l: float(roc_auc_score(y, x)) for l, y in lab.items()}
        for l, y in lab.items():
            pooled[l].append(y)
    P = {k: np.concatenate(v) for k, v in pooled.items()}
    for l in ["col", "near", "l2top5"]:
        out[l] = {"n_pos": int(P[l].sum()), "n": int(len(P[l])), "base_rate": float(P[l].mean()), "metrics": {}}
        for f in feats:
            rng = [per[n][f][l] for n in per]
            out[l]["metrics"][f] = dict(auroc=float(roc_auc_score(P[l], P[f])),
                                        pr_auc=float(average_precision_score(P[l], P[f])),
                                        auroc_min=min(rng), auroc_max=max(rng))
    return out


def time_risk(dump_dir, n_frames, agent_thresh=0.3, dt=0.1):
    """Wall time (single process, numpy) of the deployable computation used online:
    trigger = min distance of the chosen trajectory; if triggered, col_prob / min_dist / max_accel
    of the 6 commanded candidates. Both parts are timed for every frame."""
    from risk_eval import rect_dist
    from risk_metrics import EGO_L, EGO_W, DT_PLAN, ego_center, interp, obb_overlap, path_yaw
    recs = mmcv.load(os.path.join(dump_dir, "planning_dump.pkl"))["records"][:n_frames]
    with open("data/infos/nuscenes_infos_val.pkl", "rb") as f:
        infos = {x["token"]: x for x in pickle.load(f)["infos"]}
    td = np.arange(0, 3 + 1e-9, dt)

    def score(rec, rows):
        plan = rec["plan_traj"][rows].astype(np.float64)
        M = len(rows)
        ego = interp(np.concatenate([np.zeros((M, 1, 2)), plan], 1), np.arange(7) * DT_PLAN, td)
        yaw = path_yaw(ego, np.pi / 2)
        ec = ego_center(ego, yaw)
        ag = rec["agents"]
        k = ag["cls_scores"] >= agent_thresh
        if not k.any():
            return np.zeros(M), np.full(M, np.inf)
        box = ag["boxes"][k].astype(np.float64)
        tr = ag["trajs"][k].astype(np.float64)
        p = ag["trajs_score"][k].astype(np.float64)
        p /= np.maximum(p.sum(-1, keepdims=True), 1e-9)
        N = len(box)
        cur = np.broadcast_to(box[:, None, None, :2], (N, 6, 1, 2))
        axy = interp(np.concatenate([cur, tr], 2), np.arange(13) * DT_PLAN, td)
        ayaw = path_yaw(axy, np.broadcast_to(box[:, None, 6], (N, 6)))
        ov = obb_overlap(ec[:, None, None], EGO_W, EGO_L, yaw[:, None, None], axy[None],
                         box[None, :, 3, None, None], box[None, :, 4, None, None], ayaw[None])
        colp = (ov.any(-1) * p[None]).sum(-1).max(-1)
        top = p.argmax(-1)
        g = rect_dist(ec[:, None], EGO_W, EGO_L, yaw[:, None], axy[np.arange(N), top][None],
                      box[None, :, 3, None], box[None, :, 4, None], ayaw[np.arange(N), top][None])
        return colp, g.min(axis=(-1, -2))

    def accel(rec, rows, v0):
        P = np.concatenate([np.zeros((len(rows), 1, 2)), rec["plan_traj"][rows].astype(np.float64)], 1)
        v = np.diff(P, axis=1) / DT_PLAN
        a = np.concatenate([(v[:, :1] - np.array([0, v0])) / (DT_PLAN / 2), np.diff(v, axis=1) / DT_PLAN], 1)
        return np.linalg.norm(a, axis=-1).max(-1)

    t_trig, t_full = [], []
    for rec in recs:
        c, s = rec["cmd"], rec["sel_mode"]
        v0 = float(infos[rec["token"]]["ego_status"][6])
        t0 = time.perf_counter()
        score(rec, [c * 6 + s])
        t1 = time.perf_counter()
        score(rec, list(range(c * 6, c * 6 + 6)))
        accel(rec, list(range(c * 6, c * 6 + 6)), v0)
        t2 = time.perf_counter()
        t_trig.append(t1 - t0); t_full.append(t2 - t1)
    f = lambda x: dict(mean_ms=float(np.mean(x) * 1e3), p50_ms=float(np.median(x) * 1e3), p95_ms=float(np.quantile(x, .95) * 1e3))
    return dict(n_frames=len(recs), trigger=f(t_trig), rescore6=f(t_full),
                note="CPU, numpy, one process (OMP threads default), agents cls >= 0.3, 0.1 s grid")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", action="append", required=True, help="NAME=work_dirs/risk_eval/NAME")
    ap.add_argument("--out", required=True)
    ap.add_argument("--time-frames", type=int, default=500)
    args = ap.parse_args()
    runs = {n: load_run(p) for n, p in (x.split("=", 1) for x in args.run)}

    # fixed 2-fold scene split (same for every model: all dumps share the val order)
    scenes = sorted(set(next(iter(runs.values()))["scene"]))
    perm = np.random.RandomState(0).permutation(len(scenes))
    fold_of = {scenes[j]: int(k < len(scenes) // 2) for k, j in enumerate(perm)}
    res = dict(split=dict(n_scenes=len(scenes), fold_sizes=[sum(1 for v in fold_of.values() if v == f) for f in (0, 1)]),
               grid={k: [float(x) for x in v] for k, v in GRID.items()}, d_cap=D_CAP, l2_slack=L2_SLACK,
               auroc_cmd_row=auroc_rows(runs), models={})
    for name, A in runs.items():
        fold = np.array([fold_of[s] for s in A["scene"]])
        full_all = np.ones(len(fold), bool)
        ch = A["sel"].copy()
        trig_all = np.zeros(len(fold), bool)
        params = {}
        for f in (0, 1):
            p = tune(A, fold == 1 - f)                   # tune on the other fold
            c, t = choose(A, fold == f, **p)
            ch[fold == f], trig_all[fold == f] = c[fold == f], t[fold == f]
            params[f"eval_fold{f}"] = {k: float(v) for k, v in p.items()}
        base = evaluate(A, A["sel"], full_all)
        plan2 = evaluate(A, ch, full_all, trig_all)
        res["models"][name] = dict(baseline=base, plan2=plan2, params=params,
                                   fold_sizes=[int((fold == f).sum()) for f in (0, 1)])
        print(name, "base col %.4f%% L2 %.4f | plan2 col %.4f%% L2 %.4f trig %.3f chg %.3f" % (
            base["obj_box_col"] * 100, base["L2"], plan2["obj_box_col"] * 100, plan2["L2"],
            plan2["trigger_rate"], plan2["changed_rate"]), params)
    first = args.run[0].split("=", 1)[1]
    res["timing"] = time_risk(first, args.time_frames)
    print(res["timing"])
    os.makedirs(args.out, exist_ok=True)
    with open(os.path.join(args.out, "reselect.json"), "w") as f:
        json.dump(res, f, indent=1, default=float)


if __name__ == "__main__":
    main()
