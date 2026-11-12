"""Compare per-sample planning collisions (obj_box_col rule) between models, from dump_planning.py dumps.

A sample "collides" when, at any of the 6 future steps, the selected trajectory's ego box overlaps a GT
future box while the GT ego trajectory does not (planning_eval.PlanningMetric obj_box_col rule).
Only valid samples (all 6 GT future steps present) count, as in the official metric.

  PYTHONPATH=. python tools/feasibility/collision_compare.py \
      --dump D=work_dirs/risk_eval/D_L6_iter21096/planning_dump.pkl --dump C=... --dump E=... \
      --ref D --out work_dirs/risk_eval/collision_compare.json
"""
import argparse
import json
import os
from collections import Counter, defaultdict

import mmcv
import numpy as np


def scene_names(dataroot):
    path = os.path.join(dataroot, "v1.0-trainval", "scene.json")
    with open(path) as f:
        return {s["token"]: s["name"] for s in json.load(f)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dump", action="append", required=True, help="NAME=path/to/planning_dump.pkl")
    ap.add_argument("--ref", required=True, help="reference model name (e.g. D)")
    ap.add_argument("--out", required=True)
    ap.add_argument("--dataroot", default="data/nuscenes")
    args = ap.parse_args()

    names = scene_names(args.dataroot)
    models, metric = {}, {}
    for item in args.dump:
        n, p = item.split("=", 1)
        d = mmcv.load(p)
        models[n] = d["records"]
        metric[n] = d["meta"]["metric"]
    ref = args.ref
    tokens = [r["token"] for r in models[ref]]
    for n, recs in models.items():
        assert [r["token"] for r in recs] == tokens, n

    recs0 = models[ref]
    valid = np.array([r["valid"] for r in recs0])
    for n, recs in models.items():
        assert (np.array([r["valid"] for r in recs]) == valid).all()
    scene_of = [names.get(r["scene_token"], r["scene_token"]) for r in recs0]
    n_scene_valid = Counter(s for s, v in zip(scene_of, valid) if v)

    col = {n: np.array([r["valid"] and bool(r["col"].any()) for r in recs]) for n, recs in models.items()}
    col_steps = {n: np.array([r["col"].sum() if r["valid"] else 0 for r in recs]) for n, recs in models.items()}
    gt_col = np.array([r["valid"] and bool(r["gt_col"].any()) for r in recs0])

    out = dict(n_samples=len(tokens), n_valid=int(valid.sum()), n_scenes=len(n_scene_valid),
               gt_traj_collides_samples=int(gt_col.sum()), models={})
    for n in models:
        per_scene = Counter(s for s, c in zip(scene_of, col[n]) if c)
        out["models"][n] = dict(
            collide_samples=int(col[n].sum()),
            collide_steps=int(col_steps[n].sum()),
            obj_box_col_from_dump=metric[n]["col"],
            scenes_with_collision=len(per_scene),
            per_scene=dict(per_scene.most_common()),
            per_scene_hist=dict(sorted(Counter(per_scene.values()).items())),
        )

    def listing(mask):
        by = defaultdict(list)
        for i in np.where(mask)[0]:
            by[scene_of[i]].append(dict(idx=int(i), token=tokens[i], cmd=int(recs0[i]["cmd"]),
                                        steps={n: int(col_steps[n][i]) for n in models}))
        return dict(n_samples=int(mask.sum()), n_scenes=len(by),
                    scenes={s: by[s] for s in sorted(by, key=lambda s: (-len(by[s]), s))})

    others = [n for n in models if n != ref]
    only = {}
    for n in others:
        only[f"{n}_only_not_{ref}"] = listing(col[n] & ~col[ref])
        only[f"{ref}_only_not_{n}"] = listing(col[ref] & ~col[n])
    if len(others) >= 2:
        any_o = np.any([col[n] for n in others], axis=0)
        all_o = np.all([col[n] for n in others], axis=0)
        only[f"any_of_{'_'.join(others)}_not_{ref}"] = listing(any_o & ~col[ref])
        only[f"all_of_{'_'.join(others)}_not_{ref}"] = listing(all_o & ~col[ref])
        only[f"all_models"] = listing(np.all([col[n] for n in models], axis=0))
    out["sets"] = only
    out["scene_sample_counts_valid"] = dict(n_scene_valid)
    out["note"] = ("collide = valid sample whose selected trajectory hits a GT box at any step while the GT "
                   "trajectory does not; evaluation is nondeterministic, so sample-level sets include eval noise")
    mmcv.mkdir_or_exist(os.path.dirname(os.path.abspath(args.out)))
    with open(args.out, "w") as f:
        json.dump(out, f, indent=1)
    s = {k: v for k, v in out.items() if k not in ("sets", "scene_sample_counts_valid", "models")}
    s["models"] = {n: {k: v for k, v in m.items() if k != "per_scene"} for n, m in out["models"].items()}
    s["sets"] = {k: (v["n_samples"], v["n_scenes"]) for k, v in only.items()}
    print(json.dumps(s, indent=1))


if __name__ == "__main__":
    main()
