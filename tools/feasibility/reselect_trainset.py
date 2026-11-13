"""Step 6, plan 2: choose the v2 re-selection parameters once on a train-set subset, evaluate once
on the full val set. CPU only (the train dumps are made beforehand with dump_planning.py --ann-file).

Fixed in advance:
  rule       v2 (reselect_v3.choose(variant="v2"): trigger pred_min_dist < tau, drop candidates
             with pred_max_accel > chosen + 1.0 m/s^2, lowest pred_col_prob then largest
             pred_min_dist, keep the model's choice on ties)
  grid       detection-score threshold {0.1, 0.2, 0.3} x tau {0.25, 0.5, 1.0, 1.5} m
  selection  on the train subset, all 5 checkpoints together: among the grid points whose mean L2
             is at most the mean baseline L2 + 0.02 m, the lowest mean obj_box_col (ties: lower
             mean L2, then fewer changes)
  val        the chosen point is applied once to the full val set; per checkpoint obj_box_col,
             removed / introduced collisions, exact McNemar, scene-cluster bootstrap 95 % CI
             (B = 1000, seed 0), L2, near-miss, trigger / change rates
pred_col_prob / pred_min_dist at 0.1 and 0.2 are recomputed exactly (reselect_v2.online_scores);
0.3 is risk_eval.py's value.

  PYTHONPATH=. python tools/feasibility/reselect_trainset.py \
      --train D_seed0=work_dirs/risk_eval/train150/D_seed0 ... --val D_seed0=work_dirs/risk_eval/D_seed0 ... \
      --train-infos work_dirs/risk_eval/train150/nuscenes_infos_train150.pkl --out work_dirs/risk_eval/train150
"""
import argparse
import itertools
import json
import os
import pickle
from multiprocessing import Pool

import mmcv
import numpy as np

import reselect_v3 as V3
from reselect import evaluate, load_run
from reselect_v2 import L2_SLACK, mcnemar

THRS = [0.1, 0.2, 0.3]
TAUS = [0.25, 0.5, 1.0, 1.5]
B = 1000


def with_thresholds(path, A, infos_path, workers):
    out = {0.3: A}
    recs = mmcv.load(os.path.join(path, "planning_dump.pkl"))["records"]
    with open(infos_path, "rb") as f:
        v0 = {x["token"]: x["ego_status"][6] for x in pickle.load(f)["infos"]}
    V3.G.update(recs=recs, v0=v0, td=np.arange(0, 3 + 1e-9, 0.1))
    with Pool(workers) as pool:
        for thr in [t for t in THRS if t != 0.3]:
            res = pool.map(V3._scores, [(int(i), thr) for i in A["idx"]], chunksize=32)
            out[thr] = dict(A, colp=np.stack([c for c, _ in res]), md=np.stack([m for _, m in res]))
    V3.G.clear()
    return out


def load_set(spec, infos_path, workers):
    runs = {}
    for item in spec:
        n, p = item.split("=", 1)
        A = load_run(p)
        runs[n] = with_thresholds(p, A, infos_path, workers)
        print("loaded", n, p, flush=True)
    return runs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--train", action="append", required=True)
    ap.add_argument("--val", action="append", required=True)
    ap.add_argument("--train-infos", required=True)
    ap.add_argument("--val-infos", default="data/infos/nuscenes_infos_val.pkl")
    ap.add_argument("--out", required=True)
    ap.add_argument("--workers", type=int, default=12)
    args = ap.parse_args()
    tr = load_set(args.train, args.train_infos, args.workers)
    names = list(tr)

    grid = []
    for thr, tau in itertools.product(THRS, TAUS):
        cols, l2s, l2b, chg, base_cols = [], [], [], [], []
        for n in names:
            A = tr[n][thr]
            m = np.ones(len(A["sel"]), bool)
            ch, trig = V3.choose(A, m, tau, "v2")
            e, b = evaluate(A, ch, m, trig), evaluate(A, A["sel"], m)
            cols.append(e["obj_box_col"]); l2s.append(e["L2"]); chg.append(e["changed_rate"])
            l2b.append(b["L2"]); base_cols.append(b["obj_box_col"])
        grid.append(dict(thr=thr, tau=tau, mean_obj_box_col=float(np.mean(cols)), mean_base_obj_box_col=float(np.mean(base_cols)),
                         mean_L2=float(np.mean(l2s)), mean_base_L2=float(np.mean(l2b)), mean_changed=float(np.mean(chg)),
                         feasible=bool(np.mean(l2s) <= np.mean(l2b) + L2_SLACK),
                         per_model_obj_box_col={n: float(c) for n, c in zip(names, cols)}))
    feas = [g for g in grid if g["feasible"]]
    best = min(feas, key=lambda g: (g["mean_obj_box_col"], g["mean_L2"], g["mean_changed"]))
    thr, tau = best["thr"], best["tau"]
    print("chosen on train:", thr, tau, flush=True)
    train_base = {n: evaluate(tr[n][0.3], tr[n][0.3]["sel"], np.ones(len(tr[n][0.3]["sel"]), bool)) for n in names}
    del tr

    va = load_set(args.val, args.val_infos, args.workers)
    scenes = sorted(set(va[names[0]][0.3]["scene"]))
    sid = {s: i for i, s in enumerate(scenes)}
    draws = np.random.RandomState(0).randint(0, len(scenes), size=(B, len(scenes)))
    W = np.stack([np.bincount(d, minlength=len(scenes)) for d in draws]).astype(np.float64)
    models, mean_d = {}, np.zeros(B)
    for n in names:
        A = va[n][thr]
        full = np.ones(len(A["sel"]), bool)
        ch, trig = V3.choose(A, full, tau, "v2")
        b, e = evaluate(A, A["sel"], full), evaluate(A, ch, full, trig)
        r = np.arange(len(ch))
        mc = mcnemar(A["colt"][r, A["sel"]].any(-1), A["colt"][r, ch].any(-1))
        mn = mcnemar(A["near"][r, A["sel"]].astype(bool), A["near"][r, ch].astype(bool))
        d, s_rem, s_int = V3.bootstrap(A, ch, np.array([sid[s] for s in A["scene"]]), len(scenes), W)
        mean_d += d / len(names)
        models[n] = dict(baseline=b, plan2=e, mcnemar_col=mc, mcnemar_near=mn, delta=e["obj_box_col"] - b["obj_box_col"],
                         ci95=np.percentile(d, [2.5, 97.5]).tolist(), scenes_removed=s_rem, scenes_introduced=s_int)
        print(f"{n}: col {b['obj_box_col']*100:.3f}->{e['obj_box_col']*100:.3f}% rem/int {mc['removed']}/{mc['introduced']} p={mc['p_value']:.2g} "
              f"CI [{np.percentile(d,2.5)*100:+.3f},{np.percentile(d,97.5)*100:+.3f}] trig {e['trigger_rate']:.3f} chg {e['changed_rate']:.3f}", flush=True)
    res = dict(thresholds=THRS, taus=TAUS, l2_slack=L2_SLACK, chosen=dict(thr=thr, tau=tau), train_grid=grid,
               train_baseline=train_base, val=dict(models=models, mean_delta=float(np.mean([m["delta"] for m in models.values()])),
                                                  mean_ci95=np.percentile(mean_d, [2.5, 97.5]).tolist()))
    os.makedirs(args.out, exist_ok=True)
    with open(os.path.join(args.out, "reselect_trainset.json"), "w") as f:
        json.dump(res, f, indent=1, default=float)


if __name__ == "__main__":
    main()
