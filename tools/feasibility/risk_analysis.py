"""Step 6 (risk score), part 3: how well each risk metric flags bad outcomes, and model comparison.
Reads risk_eval.py outputs (one per model/checkpoint). CPU only.

Two evaluation units:
  selected    one row per valid sample: the trajectory the model actually outputs
  candidates  18 rows per valid sample: every candidate of the [18, 6, 2] output
Labels (fixed in advance; computed per model before pooling):
  col         GT collision (obj_box_col rule; selected uses the official per-sample flag)
  near        GT min distance < 1.0 m (risk_eval --near-thresh; includes collisions)
  l2top5      L2 (ADE over 6 steps) in the top 5 % of that model's pool for that unit
Each metric has a fixed orientation (+1: larger = riskier, -1: smaller = riskier), set below before
looking at any result. AUROC < 0.5 therefore means the metric points the other way.

  PYTHONPATH=. python tools/feasibility/risk_analysis.py --run D_seed0=work_dirs/risk_eval/D_seed0/risk_eval.pkl ... \
      --out work_dirs/risk_eval/analysis
"""
import argparse
import json
import os

import mmcv
import numpy as np
from scipy.stats import spearmanr
from sklearn.metrics import average_precision_score, roc_auc_score

# name: (orientation, family, cap used for means)   family: pred = deployable, gt = validation
METRICS = {
    "pred_ttc": (-1, "pred", 3.5), "pred_min_dist": (-1, "pred", 50), "pred_pet": (-1, "pred", 3.5),
    "pred_drac": (+1, "pred", None), "pred_col_prob": (+1, "pred", None),
    "pred_vru_ttc": (-1, "pred", 3.5), "pred_vru_min_dist": (-1, "pred", 50),
    "pred_max_accel": (+1, "pred", None), "pred_max_decel": (+1, "pred", None),
    "pred_max_jerk": (+1, "pred", None), "pred_max_yaw_rate": (+1, "pred", None),
    "pred_path_len": (+1, "pred", None), "pred_fwd_disp": (+1, "pred", None),
    "pred_cmd_violation": (+1, "pred", None), "pred_cmd_margin": (-1, "pred", None),
    "pred_row_entropy": (+1, "pred", None), "pred_row_margin_neg": (+1, "pred", None),
    "pred_row_spread": (+1, "pred", None), "pred_cand_nll": (+1, "pred", None), "pred_cand_dev": (+1, "pred", None),
    "pred_cross_bnd": (+1, "pred", None), "pred_bnd_margin": (-1, "pred", 50),
    "pred_line_dist": (-1, "pred", 50), "pred_lane_center_off": (+1, "pred", None),
    "pred_lane_head_err": (+1, "pred", None),
    "gt_ttc": (-1, "gt", 3.5), "gt_min_dist": (-1, "gt", 50), "gt_pet": (-1, "gt", 3.5),
    "gt_drac": (+1, "gt", None), "gt_vru_ttc": (-1, "gt", 3.5), "gt_vru_min_dist": (-1, "gt", 50),
    "gt_offroad": (+1, "gt", None), "gt_lane_dist": (+1, "gt", 50), "gt_lane_head_err": (+1, "gt", None),
}
TAUTOLOGY = {"near": {"gt_min_dist"}, "col": {"gt_ttc", "gt_min_dist", "gt_pet"}}  # label is defined by these
EVENTS = {   # model comparison: rate of samples where the event happens (selected trajectory)
    "col": None, "near": None,
    "pred_ttc<=3s": ("pred_ttc", lambda v: np.isfinite(v)), "pred_pet<=1s": ("pred_pet", lambda v: v <= 1.0),
    "pred_drac>3.35": ("pred_drac", lambda v: v > 3.35), "pred_cross_bnd": ("pred_cross_bnd", lambda v: v > 0),
    "pred_cmd_violation": ("pred_cmd_violation", lambda v: v > 0),
    "gt_ttc<=3s": ("gt_ttc", lambda v: np.isfinite(v)), "gt_pet<=1s": ("gt_pet", lambda v: v <= 1.0),
    "gt_drac>3.35": ("gt_drac", lambda v: v > 3.35), "gt_offroad": ("gt_offroad", lambda v: v > 0),
}


def load(path):
    d = mmcv.load(path)
    risk, sel, valid = d["risk"], np.array(d["sel"]), np.array(d["valid"])
    vi = np.where(valid)[0]
    out = {"selected": {}, "candidates": {}}
    for m in METRICS:
        key = "gt_offroad_t" if m == "gt_offroad" else m
        a = np.stack([risk[i][key] for i in vi]).astype(np.float64)
        if m == "gt_offroad":
            a = a.any(-1).astype(float)
        out["candidates"][m] = a.ravel()
        out["selected"][m] = a[np.arange(len(vi)), sel[vi]]
    gcol = np.stack([risk[i]["gt_col"] for i in vi])
    near = np.stack([risk[i]["gt_near"] for i in vi])
    l2 = np.stack([risk[i]["l2_ade"] for i in vi])
    offi = np.array(d["official_col"])[vi]
    l2s = l2[np.arange(len(vi)), sel[vi]]
    out["selected"].update(col=offi, near=near[np.arange(len(vi)), sel[vi]], l2top5=l2s >= np.quantile(l2s, 0.95),
                           _l2=l2s)
    lc = l2.ravel()
    out["candidates"].update(col=gcol.ravel(), near=near.ravel(), l2top5=lc >= np.quantile(lc, 0.95), _l2=lc)
    out["check_col_agreement_selected"] = float((gcol[np.arange(len(vi)), sel[vi]] == offi).mean())
    out["gt_boxes_match"] = d["meta"]["gt_boxes_match"]
    out["n_valid"] = len(vi)
    return out


def score_table(rows, labels=("col", "near", "l2top5")):
    res = {}
    for lab in labels:
        y = rows[lab].astype(bool)
        res[lab] = {"n_pos": int(y.sum()), "n": int(len(y)), "base_rate": float(y.mean()), "metrics": {}}
        for m, (sign, fam, _) in METRICS.items():
            x = rows[m]
            ok = ~np.isnan(x)
            s = sign * np.nan_to_num(x[ok], posinf=1e6, neginf=-1e6)
            if y[ok].all() or not y[ok].any():
                continue
            res[lab]["metrics"][m] = dict(auroc=float(roc_auc_score(y[ok], s)),
                                          pr_auc=float(average_precision_score(y[ok], s)),
                                          family=fam, tautology=m in TAUTOLOGY.get(lab, set()))
    return res


def concat(runs, unit):
    keys = list(METRICS) + ["col", "near", "l2top5"]
    return {k: np.concatenate([r[unit][k] for r in runs.values()]) for k in keys}


def means(rows):
    o = {}
    for m, (sign, fam, cap) in METRICS.items():
        x = rows[m].copy()
        if cap is not None:
            x = np.minimum(x, cap)
        o[m] = float(np.nanmean(x))
    o["l2_ade"] = float(rows["_l2"].mean())
    for ev, spec in EVENTS.items():
        if spec is None:
            o[f"rate_{ev}"] = float(rows[ev].mean())
        else:
            o[f"rate_{ev}"] = float(spec[1](rows[spec[0]]).mean())
    return o


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", action="append", required=True, help="NAME=path/to/risk_eval.pkl")
    ap.add_argument("--out", required=True, help="output directory")
    args = ap.parse_args()
    runs = {}
    for item in args.run:
        n, p = item.split("=", 1)
        runs[n] = load(p)
    os.makedirs(args.out, exist_ok=True)
    res = {"checks": {n: dict(n_valid=r["n_valid"], col_agreement_selected=r["check_col_agreement_selected"],
                              gt_boxes_match=r["gt_boxes_match"]) for n, r in runs.items()}}
    for unit in ["selected", "candidates"]:
        res[unit] = {"pooled": score_table(concat(runs, unit)),
                     "per_model": {n: score_table(r[unit]) for n, r in runs.items()}}
        names = [m for m in METRICS if METRICS[m][1] == "pred"]
        rows = concat(runs, unit)
        X = np.stack([np.nan_to_num(METRICS[m][0] * rows[m], nan=0, posinf=1e6, neginf=-1e6) for m in names], 1)
        rho = spearmanr(X).correlation
        res[unit]["spearman_pred"] = {"names": names, "rho": np.round(rho, 3).tolist()}
        gn = [m for m in METRICS if METRICS[m][1] == "gt"]
        Xg = np.stack([np.nan_to_num(METRICS[m][0] * rows[m], nan=0, posinf=1e6, neginf=-1e6) for m in gn], 1)
        rho_pg = spearmanr(np.concatenate([X, Xg], 1)).correlation[:len(names), len(names):]
        res[unit]["spearman_pred_vs_gt"] = {"pred": names, "gt": gn, "rho": np.round(rho_pg, 3).tolist()}
    res["model_means_selected"] = {n: means(r["selected"]) for n, r in runs.items()}
    with open(os.path.join(args.out, "risk_analysis.json"), "w") as f:
        json.dump(res, f, indent=1)
    print(json.dumps(res["checks"], indent=1))


if __name__ == "__main__":
    main()
