"""Step 6, plan 2 v2 (shared tau): scene-level view of removed / introduced collisions and a
scene-cluster bootstrap CI of the obj_box_col reduction. CPU only.

Uses exactly the shared-tau choices of reselect_v2.py (same split, same tuning, tau per fold read
back from its JSON and checked by re-tuning). The bootstrap resamples the 150 val scenes with
replacement (B = 1000, seed 0) and recomputes obj_box_col (planning_eval formula, samples weighted by
how often their scene was drawn) for baseline and plan 2 on the same draw. tau is held fixed, so the
interval covers scene sampling, not the tuning step.

  PYTHONPATH=. python tools/feasibility/reselect_v2_scene.py --run D_seed0=work_dirs/risk_eval/D_seed0 ... \
      --v2 work_dirs/risk_eval/reselect_v2/reselect_v2.json --out work_dirs/risk_eval/reselect_v2
"""
import argparse
import json
import os

import numpy as np

from reselect import load_run
from reselect_v2 import choose, tune_shared

B = 1000


def obj_box_col_weighted(per_scene_step_sum, per_scene_n, w):
    per_t = (w[:, None] * per_scene_step_sum).sum(0) / (w * per_scene_n).sum()
    cum = np.cumsum(per_t) / np.arange(1, 7)
    return cum[[1, 3, 5]].mean()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", action="append", required=True)
    ap.add_argument("--v2", required=True, help="reselect_v2.json (for the tuned tau per fold)")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    paths = dict(x.split("=", 1) for x in args.run)
    runs = {n: load_run(p) for n, p in paths.items()}
    scenes = sorted(set(next(iter(runs.values()))["scene"]))
    perm = np.random.RandomState(0).permutation(len(scenes))
    fold_of = {scenes[j]: int(k < len(scenes) // 2) for k, j in enumerate(perm)}
    folds = {n: np.array([fold_of[s] for s in A["scene"]]) for n, A in runs.items()}
    v2 = json.load(open(args.v2))
    tau = {f: float(v2["tau_tuned"][f"eval_fold{f}"]) for f in (0, 1)}
    assert all(tune_shared(runs, folds, 1 - f) == tau[f] for f in (0, 1)), "tau differs from reselect_v2.json"

    rng = np.random.RandomState(0)
    draws = rng.randint(0, len(scenes), size=(B, len(scenes)))
    W = np.stack([np.bincount(d, minlength=len(scenes)) for d in draws]).astype(np.float64)   # [B, S]
    sidx = {s: i for i, s in enumerate(scenes)}
    res = dict(tau=tau, B=B, n_scenes=len(scenes), models={})
    pooled_delta = np.zeros(B)
    for n, A in runs.items():
        ch = A["sel"].copy()
        for f in (0, 1):
            m = folds[n] == f
            c, _ = choose(A, m, tau[f])
            ch[m] = c[m]
        r = np.arange(len(ch))
        cb, cn = A["colt"][r, A["sel"]], A["colt"][r, ch]                   # [n, 6] per-step collisions
        fb, fn = cb.any(-1), cn.any(-1)
        si = np.array([sidx[s] for s in A["scene"]])
        S = len(scenes)
        sum_b = np.zeros((S, 6)); sum_n = np.zeros((S, 6)); cnt = np.zeros(S)
        np.add.at(sum_b, si, cb); np.add.at(sum_n, si, cn); np.add.at(cnt, si, 1)
        rem = np.bincount(si, weights=(fb & ~fn), minlength=S)
        intro = np.bincount(si, weights=(~fb & fn), minlength=S)
        net = intro - rem
        ones = np.ones(S)
        point = obj_box_col_weighted(sum_n, cnt, ones) - obj_box_col_weighted(sum_b, cnt, ones)
        delta = np.array([obj_box_col_weighted(sum_n, cnt, w) - obj_box_col_weighted(sum_b, cnt, w) for w in W])
        pooled_delta += delta / len(runs)
        lo, hi = np.percentile(delta, [2.5, 97.5])
        changed_scenes = [scenes[i] for i in np.where((rem > 0) | (intro > 0))[0]]
        res["models"][n] = dict(
            scenes_with_removed=int((rem > 0).sum()), scenes_with_introduced=int((intro > 0).sum()),
            scenes_with_both=int(((rem > 0) & (intro > 0)).sum()),
            scenes_net_better=int((net < 0).sum()), scenes_net_worse=int((net > 0).sum()),
            samples_removed=int(rem.sum()), samples_introduced=int(intro.sum()),
            delta_obj_box_col=float(point), ci95=[float(lo), float(hi)],
            p_boot_delta_ge_0=float((delta >= 0).mean()),
            per_scene={scenes[i]: dict(removed=int(rem[i]), introduced=int(intro[i])) for i in range(S)
                       if rem[i] or intro[i]},
            n_changed_scenes=len(changed_scenes))
        print(f"{n}: scenes removed {res['models'][n]['scenes_with_removed']} introduced {res['models'][n]['scenes_with_introduced']} "
              f"(both {res['models'][n]['scenes_with_both']}; net better {res['models'][n]['scenes_net_better']} worse {res['models'][n]['scenes_net_worse']}) "
              f"delta {point*100:+.4f}%p CI [{lo*100:+.4f}, {hi*100:+.4f}] P(delta>=0) {res['models'][n]['p_boot_delta_ge_0']:.3f}")
    lo, hi = np.percentile(pooled_delta, [2.5, 97.5])
    res["mean_of_checkpoints"] = dict(ci95=[float(lo), float(hi)], p_boot_delta_ge_0=float((pooled_delta >= 0).mean()))
    print("mean over checkpoints CI", lo * 100, hi * 100, res["mean_of_checkpoints"]["p_boot_delta_ge_0"])
    os.makedirs(args.out, exist_ok=True)
    with open(os.path.join(args.out, "reselect_v2_scene.json"), "w") as f:
        json.dump(res, f, indent=1)


if __name__ == "__main__":
    main()
