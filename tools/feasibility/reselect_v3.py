"""Step 6, plan 2 (v3): reselect_v2.py plus a guard against switching into a predicted collision.
CPU only, from dump_planning.py dumps (+ val infos for the ego speed). No training.

Variants (all share v2's trigger: pred_min_dist of the model's choice < tau, and v2's acceleration
filter: drop candidates whose pred_max_accel exceeds the model's choice + ACC_X):
  v2           lowest pred_col_prob, then largest pred_min_dist (reselect_v2.py)
  colp0        switch only to a candidate with pred_col_prob == 0: among those, the largest
               pred_min_dist; if the model's choice already has col_prob 0, only a strictly larger
               min_dist replaces it                                                  (main)
  allcol_keep  v2, but do not switch when all 6 commanded candidates have pred_col_prob > 0
               (the built-in rescore's rule)
For every variant and every detection-score threshold (risk-score agents with cls >= thr; 0.3 is
the risk_eval.py default) one tau is tuned jointly for the 5 checkpoints on one fold and applied to
the other fold (reselect_v2.tune_shared objective: mean obj_box_col s.t. mean L2 <= base + 0.02 m),
and per checkpoint we report obj_box_col, L2, near-miss, trigger / change rates, removed /
introduced collision samples, exact McNemar and a scene-cluster bootstrap 95 % CI (B = 1000, seed 0,
tau held fixed) of the obj_box_col change.
pred_col_prob / pred_min_dist are recomputed per threshold with reselect_v2.online_scores (exact,
pruned); at 0.3 they are checked against risk_eval.py.

  PYTHONPATH=. python tools/feasibility/reselect_v3.py --run D_seed0=work_dirs/risk_eval/D_seed0 ... \
      --out work_dirs/risk_eval/reselect_v3
"""
import argparse
import json
import os
import pickle
from multiprocessing import Pool

import mmcv
import numpy as np

from reselect import evaluate, load_run
from reselect_v2 import ACC_X, L2_SLACK, TAU_GRID, mcnemar, online_scores
from reselect_v2_scene import obj_box_col_weighted

VARIANTS = ["colp0", "allcol_keep", "v2"]
THRESHOLDS = [0.1, 0.2, 0.3]
B = 1000
G = {}


def choose(A, m, tau, variant):
    n = len(A["sel"])
    r = np.arange(n)
    sel = A["sel"]
    colp, md, acc = A["colp"], np.minimum(A["md"], 1e6), A["acc"]
    trig = (md[r, sel] < tau) & m
    allowed = acc <= acc[r, sel][:, None] + ACC_X
    if variant == "colp0":
        cand = allowed & (colp == 0)
        mdc = np.where(cand, md, -np.inf)
        best = mdc.argmax(-1)
        has = cand[r, best]
        better = has & ((colp[r, sel] > 0) | (mdc[r, best] > md[r, sel]))
        new = np.where(better, best, sel)
    else:
        cp = np.where(allowed, colp, np.inf)
        mm = np.where(allowed, md, -np.inf)
        tie = np.isclose(cp, cp.min(-1, keepdims=True), rtol=0, atol=1e-9)
        mt = np.where(tie, mm, -np.inf)
        best = mt.argmax(-1)
        keep = tie[r, sel] & (mm[r, sel] >= mt[r, best])
        new = np.where(keep, sel, best)
        if variant == "allcol_keep":
            new = np.where((colp > 0).all(-1), sel, new)
    return np.where(trig, new, sel), trig


def tune_shared(runs, folds, f_tune, variant):
    best = None
    for tau in TAU_GRID:
        cols, l2s, l2b, chg = [], [], [], []
        for n, A in runs.items():
            m = folds[n] == f_tune
            ch, trig = choose(A, m, tau, variant)
            e = evaluate(A, ch, m, trig)
            cols.append(e["obj_box_col"]); l2s.append(e["L2"]); chg.append(e["changed_rate"])
            l2b.append(evaluate(A, A["sel"], m)["L2"])
        if np.mean(l2s) > np.mean(l2b) + L2_SLACK:
            continue
        key = (np.mean(cols), np.mean(l2s), np.mean(chg))
        if best is None or key < best[0]:
            best = (key, tau)
    return best[1]


def bootstrap(A, ch, sidx, n_scenes, W):
    r = np.arange(len(ch))
    cb, cn = A["colt"][r, A["sel"]], A["colt"][r, ch]
    sb, sn, cnt = np.zeros((n_scenes, 6)), np.zeros((n_scenes, 6)), np.zeros(n_scenes)
    np.add.at(sb, sidx, cb); np.add.at(sn, sidx, cn); np.add.at(cnt, sidx, 1)
    d = np.array([obj_box_col_weighted(sn, cnt, w) - obj_box_col_weighted(sb, cnt, w) for w in W])
    fb, fn = cb.any(-1), cn.any(-1)
    rem = np.bincount(sidx, weights=(fb & ~fn), minlength=n_scenes)
    intro = np.bincount(sidx, weights=(~fb & fn), minlength=n_scenes)
    return d, int((rem > 0).sum()), int((intro > 0).sum())


def _scores(job):
    i, thr = job
    rec = G["recs"][i]
    v0 = float(G["v0"][rec["token"]])
    rows = list(range(rec["cmd"] * 6, rec["cmd"] * 6 + 6))
    colp, md, _ = online_scores(rec, rows, v0, G["td"], agent_thresh=thr, prune=True)
    return colp, md


def rescore_thresholds(path, A, workers):
    recs = mmcv.load(os.path.join(path, "planning_dump.pkl"))["records"]
    with open("data/infos/nuscenes_infos_val.pkl", "rb") as f:
        v0 = {x["token"]: x["ego_status"][6] for x in pickle.load(f)["infos"]}
    G.update(recs=recs, v0=v0, td=np.arange(0, 3 + 1e-9, 0.1))
    out = {}
    with Pool(workers) as pool:
        for thr in THRESHOLDS:
            res = pool.map(_scores, [(int(i), thr) for i in A["idx"]], chunksize=32)
            out[thr] = (np.stack([c for c, _ in res]), np.stack([m for _, m in res]))
    G.clear()
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", action="append", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--workers", type=int, default=12)
    args = ap.parse_args()
    paths = dict(x.split("=", 1) for x in args.run)
    base = {n: load_run(p) for n, p in paths.items()}
    check = {}
    per_thr = {thr: {} for thr in THRESHOLDS}
    for n, A in base.items():
        sc = rescore_thresholds(paths[n], A, args.workers)
        c3, m3 = sc[0.3]
        check[n] = dict(colp_max_abs_diff=float(np.abs(c3 - A["colp"]).max()),
                        md_max_abs_diff=float(np.abs(np.minimum(m3, 1e6) - np.minimum(A["md"], 1e6)).max()))
        for thr in THRESHOLDS:
            per_thr[thr][n] = dict(A, colp=sc[thr][0], md=sc[thr][1])
        print(n, "recomputed; check vs risk_eval at 0.3:", check[n], flush=True)

    scenes = sorted(set(next(iter(base.values()))["scene"]))
    perm = np.random.RandomState(0).permutation(len(scenes))
    fold_of = {scenes[j]: int(k < len(scenes) // 2) for k, j in enumerate(perm)}
    sid = {s: i for i, s in enumerate(scenes)}
    folds = {n: np.array([fold_of[s] for s in A["scene"]]) for n, A in base.items()}
    draws = np.random.RandomState(0).randint(0, len(scenes), size=(B, len(scenes)))
    W = np.stack([np.bincount(d, minlength=len(scenes)) for d in draws]).astype(np.float64)

    res = dict(acc_x=ACC_X, tau_grid=[float(t) for t in TAU_GRID], B=B, check_thr03=check, results={})
    for thr in THRESHOLDS:
        runs = per_thr[thr]
        for v in VARIANTS:
            tau = {f: tune_shared(runs, folds, 1 - f, v) for f in (0, 1)}
            models, mean_d = {}, np.zeros(B)
            for n, A in runs.items():
                full = np.ones(len(A["sel"]), bool)
                ch, trig = A["sel"].copy(), np.zeros(len(A["sel"]), bool)
                for f in (0, 1):
                    m = folds[n] == f
                    c, t = choose(A, m, tau[f], v)
                    ch[m], trig[m] = c[m], t[m]
                b, e = evaluate(A, A["sel"], full), evaluate(A, ch, full, trig)
                r = np.arange(len(ch))
                mc = mcnemar(A["colt"][r, A["sel"]].any(-1), A["colt"][r, ch].any(-1))
                mn = mcnemar(A["near"][r, A["sel"]].astype(bool), A["near"][r, ch].astype(bool))
                d, s_rem, s_int = bootstrap(A, ch, np.array([sid[s] for s in A["scene"]]), len(scenes), W)
                mean_d += d / len(runs)
                models[n] = dict(baseline=b, plan2=e, mcnemar_col=mc, mcnemar_near=mn,
                                 delta=e["obj_box_col"] - b["obj_box_col"], ci95=np.percentile(d, [2.5, 97.5]).tolist(),
                                 scenes_removed=s_rem, scenes_introduced=s_int)
            key = f"thr{thr}_{v}"
            res["results"][key] = dict(threshold=thr, variant=v, tau={f"eval_fold{f}": float(t) for f, t in tau.items()},
                                       models=models, mean_ci95=np.percentile(mean_d, [2.5, 97.5]).tolist(),
                                       mean_delta=float(np.mean([mm["delta"] for mm in models.values()])))
            print(key, "tau", tau, " ".join(f"{n}:{mm['plan2']['obj_box_col']*100:.3f}%({mm['mcnemar_col']['removed']}/{mm['mcnemar_col']['introduced']})"
                                            for n, mm in models.items()), flush=True)
    os.makedirs(args.out, exist_ok=True)
    with open(os.path.join(args.out, "reselect_v3.json"), "w") as f:
        json.dump(res, f, indent=1, default=float)


if __name__ == "__main__":
    main()
