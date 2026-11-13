"""Case study of the shared-tau re-selection (reselect_v2.py): frames where plan 2 introduced a GT
collision (or removed one), drawn in BEV, with a counterfactual diagnosis of the cause. CPU only.

For a case frame, the collision is the first 0.5 s step where the colliding trajectory's ego box
overlaps a GT future box while the GT ego box does not (obj_box_col rule). The GT agent hit there is
followed by instance id over steps 0..6 and matched to a detected agent (nearest centre at t = 0,
within MATCH_M; detections used by the risk score have cls >= 0.3). Diagnosis on that trajectory, on
the same dense 0.1 s grid and ego box as the risk score:
  pred      detected box size + most likely predicted mode          (what the risk score saw)
  any_mode  detected box size + any of the 6 predicted modes
  size      GT box size       + most likely predicted mode          (only the size corrected)
  traj      detected box size + GT trajectory                       (only the trajectory corrected)
  both      GT box size       + GT trajectory
"overlap" means the ego box overlaps the agent box at some time in 0..3 s. Cause labels:
  missed detection / trajectory error (traj alone explains) / box size (size alone explains) /
  both needed / either suffices / evaluation geometry (even 'both' shows no overlap).

  PYTHONPATH=. python tools/feasibility/reselect_case_study.py --run D_seed0=work_dirs/risk_eval/D_seed0 ... \
      --v2 work_dirs/risk_eval/reselect_v2/reselect_v2.json --out work_dirs/risk_eval/case_study \
      --scene scene-0563:introduced --scene scene-0016:removed:2
"""
import argparse
import json
import os
import pickle

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import mmcv
import numpy as np
from matplotlib.patches import Polygon as MPoly

from reselect import load_run
from reselect_v2 import choose
from risk_eval import gt_steps
from risk_metrics import EGO_L, EGO_W, DT_PLAN, box_corners, ego_center, eval_yaw, interp, obb_overlap, path_yaw

MATCH_M = 3.0
AGENT_THRESH = 0.3
TD = np.arange(0, 3 + 1e-9, 0.1)


def ego_dense(plan6):
    P = np.concatenate([np.zeros((1, 2)), plan6], 0)
    xy = interp(P[None], np.arange(7) * DT_PLAN, TD)[0]
    yaw = path_yaw(xy[None], np.pi / 2)[0]
    return ego_center(xy, yaw), yaw


def overlap_any(ec, eyaw, axy, w, l, ayaw):
    return bool(obb_overlap(ec, EGO_W, EGO_L, eyaw, axy, w, l, ayaw).any())


def gt_track_dense(track):
    """track: {step: box [7]} -> dense xy / yaw over TD (linear in time, held at the ends)."""
    ts = np.array(sorted(track))
    b = np.stack([track[t] for t in ts])
    t = ts * DT_PLAN
    xy = np.stack([np.interp(TD, t, b[:, j]) for j in range(2)], -1)
    yaw = np.interp(TD, t, np.unwrap(b[:, 6]))
    return xy, yaw, b


def diagnose(rec, steps, cand6, step_hit, gt_box_hit, gt_id):
    ec, eyaw = ego_dense(cand6)
    track = {t: s[0][list(s[2]).index(gt_id)] for t, s in enumerate(steps) if gt_id in list(s[2])}
    gxy, gyaw, gb = gt_track_dense(track)
    gw, gl = gt_box_hit[3], gt_box_hit[4]
    ag = rec["agents"]
    c0 = track[min(track)][:2]
    d = np.linalg.norm(ag["boxes"][:, :2] - c0, axis=-1)
    out = dict(gt_id=int(gt_id), gt_size=[float(gw), float(gl)], collision_step=int(step_hit),
               collision_time_s=float(step_hit * DT_PLAN))
    if len(d) == 0 or d.min() > MATCH_M:
        out.update(cause="missed detection (no detected box within %.0f m)" % MATCH_M, matched=None)
        return out, track, None
    j = int(d.argmin())
    sc = float(ag["cls_scores"][j])
    box = ag["boxes"][j].astype(np.float64)
    tr = ag["trajs"][j].astype(np.float64)
    p = ag["trajs_score"][j].astype(np.float64); p = p / p.sum()
    full = np.concatenate([np.broadcast_to(box[None, None, :2], (6, 1, 2)), tr], 1)
    axy = interp(full, np.arange(13) * DT_PLAN, TD)                         # [6, T, 2]
    ayaw = path_yaw(axy, np.full(6, box[6]))
    top = int(p.argmax())
    pw, pl = box[3], box[4]
    tests = dict(
        pred=overlap_any(ec, eyaw, axy[top], pw, pl, ayaw[top]),
        any_mode=any(overlap_any(ec, eyaw, axy[k], pw, pl, ayaw[k]) for k in range(6)),
        size=overlap_any(ec, eyaw, axy[top], gw, gl, ayaw[top]),
        traj=overlap_any(ec, eyaw, gxy, pw, pl, gyaw),
        both=overlap_any(ec, eyaw, gxy, gw, gl, gyaw))
    th = step_hit * DT_PLAN
    k_t = int(round(th / 0.1))
    pos_err_top = float(np.linalg.norm(axy[top, k_t] - gxy[k_t]))
    pos_err_min = float(np.linalg.norm(axy[:, k_t] - gxy[k_t], axis=-1).min())
    if sc < AGENT_THRESH:
        cause = f"detected but below the risk-score threshold (cls {sc:.2f} < {AGENT_THRESH})"
    elif tests["pred"]:
        cause = "risk score saw the overlap (should not have been chosen)"
    elif tests["traj"] and not tests["size"]:
        cause = "trajectory error"
    elif tests["size"] and not tests["traj"]:
        cause = "box size"
    elif tests["size"] and tests["traj"]:
        cause = "either suffices (trajectory or size)"
    elif tests["both"]:
        cause = "both needed (trajectory and size)"
    else:
        cause = "evaluation geometry (no overlap even with GT size and trajectory on the dense grid)"
    out.update(matched=dict(index=j, cls_score=sc, label=int(ag["labels"][j]), size=[float(pw), float(pl)],
                            center_dist_t0=float(d[j]), top_mode=top, top_mode_prob=float(p[top])),
               tests=tests, pos_err_top_at_hit=pos_err_top, pos_err_min_mode_at_hit=pos_err_min,
               size_diff=[float(gw - pw), float(gl - pl)], cause=cause)
    return out, track, (axy, top, box)


def first_hit(cand6, steps, gt_ego_col):
    yaw = eval_yaw(cand6[None])[0]
    c = ego_center(cand6[None], yaw[None])[0]
    for t in range(1, len(steps)):
        b, names, ids = steps[t]
        if not len(b) or gt_ego_col[t - 1]:          # steps where the GT ego collides are not counted
            continue
        ov = obb_overlap(c[t - 1], EGO_W, EGO_L, yaw[t - 1], b[:, :2], b[:, 3], b[:, 4], b[:, 6])
        if ov.any():
            k = int(np.where(ov)[0][0])
            return t, b[k], ids[k], names[k]
    return None


def draw(path, title, rec, steps, plans, track, pred, hit_t, lines, info_text):
    fig, ax = plt.subplots(figsize=(8, 9))
    mp = rec["map"]
    if mp["vectors"] is not None:
        for v, s, l in zip(mp["vectors"], mp["scores"], mp["labels"]):
            if s >= 0.3 and l in (1, 2):
                ax.plot(v[:, 0], v[:, 1], color="#bbbbbb" if l == 1 else "#888888", lw=1 if l == 1 else 1.6, zorder=1)
    for (b, names, ids) in steps[:1]:
        for bb in b:
            ax.add_patch(MPoly(box_corners(bb[:2], bb[3], bb[4], bb[6]), fill=False, ec="#999999", lw=0.7, zorder=2))
    gt = rec["gt_traj"]
    ax.plot(np.r_[0, gt[:, 0]], np.r_[0, gt[:, 1]], "--", color="#2ca02c", lw=1.5, label="GT ego", zorder=3)
    row = rec["cmd"] * 6
    for k in range(6):
        pt = rec["plan_traj"][row + k]
        ax.plot(np.r_[0, pt[:, 0]], np.r_[0, pt[:, 1]], color="#dddddd", lw=0.8, zorder=2)
    for name, (cand, color) in plans.items():
        ax.plot(np.r_[0, cand[:, 0]], np.r_[0, cand[:, 1]], "-o", ms=3, color=color, lw=2, label=name, zorder=5)
        if hit_t is not None:
            y = eval_yaw(cand[None])[0][hit_t - 1]
            c = ego_center(cand[None, hit_t - 1], np.array([y]))[0]
            ax.add_patch(MPoly(box_corners(c, EGO_W, EGO_L, y), fill=False, ec=color, lw=1.8, zorder=6))
    if track:
        ts = sorted(track)
        tb = np.stack([track[t] for t in ts])
        ax.plot(tb[:, 0], tb[:, 1], "-s", ms=3, color="#ff7f0e", lw=1.5, label="GT agent (hit)", zorder=4)
        if hit_t in track:
            bb = track[hit_t]
            ax.add_patch(MPoly(box_corners(bb[:2], bb[3], bb[4], bb[6]), fc="#ff7f0e", alpha=0.35, ec="#ff7f0e", zorder=4))
    if pred is not None:
        axy, top, box = pred
        for k in range(6):
            ax.plot(axy[k, :, 0], axy[k, :, 1], color="#9467bd", lw=2.2 if k == top else 0.6,
                    alpha=1 if k == top else 0.5, label="pred agent (top mode)" if k == top else None, zorder=4)
        ax.add_patch(MPoly(box_corners(box[:2], box[3], box[4], box[6]), fill=False, ec="#9467bd", ls="--", lw=1.5, zorder=4))
        if hit_t is not None:
            k_t = int(round(hit_t * DT_PLAN / 0.1))
            yy = path_yaw(axy[top][None], box[6])[0][k_t]
            ax.add_patch(MPoly(box_corners(axy[top, k_t], box[3], box[4], yy), fill=False, ec="#9467bd", lw=1.5, zorder=4))
    ax.add_patch(MPoly(box_corners(ego_center(np.zeros((1, 2)), np.array([np.pi / 2]))[0], EGO_W, EGO_L, np.pi / 2),
                       fc="#1f77b4", alpha=0.25, ec="#1f77b4", zorder=3))
    pts = [np.zeros(2), *[c for c, _ in plans.values() for c in c], *(list(track.values()) if track else [])]
    pts = np.array([p[:2] for p in pts])
    cx, cy = pts.mean(0)
    span = max(np.ptp(pts[:, 0]), np.ptp(pts[:, 1]), 20) / 2 + 6
    ax.set_xlim(cx - span, cx + span); ax.set_ylim(cy - span, cy + span)
    ax.set_aspect("equal"); ax.grid(alpha=0.3)
    ax.set_xlabel("x (m, right)"); ax.set_ylabel("y (m, forward)")
    ax.set_title(title, fontsize=10)
    ax.legend(loc="upper left", fontsize=8)
    ax.text(0.01, 0.01, info_text, transform=ax.transAxes, fontsize=7.5, va="bottom", family="monospace",
            bbox=dict(fc="white", alpha=0.85, ec="#cccccc"))
    fig.tight_layout(); fig.savefig(path, dpi=130); plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", action="append", required=True)
    ap.add_argument("--v2", required=True)
    ap.add_argument("--scene", action="append", required=True, help="NAME:introduced|removed[:max_frames]")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)
    names = {s["token"]: s["name"] for s in json.load(open("data/nuscenes/v1.0-trainval/scene.json"))}
    with open("data/infos/nuscenes_infos_val.pkl", "rb") as f:
        infos = sorted(pickle.load(f)["infos"], key=lambda e: e["timestamp"])
    tau = json.load(open(args.v2))["tau_tuned"]
    paths = dict(x.split("=", 1) for x in args.run)
    runs = {n: load_run(p) for n, p in paths.items()}
    scenes = sorted(set(next(iter(runs.values()))["scene"]))
    perm = np.random.RandomState(0).permutation(len(scenes))
    fold_of = {scenes[j]: int(k < len(scenes) // 2) for k, j in enumerate(perm)}
    wanted = [s.split(":") for s in args.scene]
    cases = []
    taken = {}
    for n, A in runs.items():
        fold = np.array([fold_of[s] for s in A["scene"]])
        ch = A["sel"].copy()
        for f in (0, 1):
            m = fold == f
            c, _ = choose(A, m, float(tau[f"eval_fold{f}"]))
            ch[m] = c[m]
        r = np.arange(len(ch))
        fb, fn = A["colt"][r, A["sel"]].any(-1), A["colt"][r, ch].any(-1)
        for w in wanted:
            sname, kind = w[0], w[1]
            limit = int(w[2]) if len(w) > 2 else None
            sel_rows = [i for i in r if names[A["scene"][i]] == sname and
                        ((kind == "introduced" and fn[i] and not fb[i]) or (kind == "removed" and fb[i] and not fn[i]))]
            key = (sname, kind)
            room = None if limit is None else max(limit - taken.get(key, 0), 0)
            pick = sel_rows if room is None else sel_rows[:room]
            taken[key] = taken.get(key, 0) + len(pick)
            cases += [(n, kind, i, A, ch) for i in pick]
    dumps = {}
    report = []
    for n, kind, i, A, ch in cases:
        if n not in dumps:
            dumps[n] = mmcv.load(os.path.join(paths[n], "planning_dump.pkl"))["records"]
        di = int(A["idx"][i])
        rec = dumps[n][di]
        steps = gt_steps(infos, di)
        row = rec["cmd"] * 6
        orig, new = rec["plan_traj"][row + A["sel"][i]], rec["plan_traj"][row + ch[i]]
        bad = new if kind == "introduced" else orig
        hit = first_hit(bad, steps, rec["gt_col"])
        diag, track, pred = (None, None, None)
        if hit is not None:
            diag, track, pred = diagnose(rec, steps, bad, hit[0], hit[1], hit[2])
            diag["gt_class"] = str(hit[3])
        risk = dict(orig=dict(col_prob=float(A["colp"][i, A["sel"][i]]), min_dist=float(A["md"][i, A["sel"][i]]),
                              max_accel=float(A["acc"][i, A["sel"][i]])),
                    new=dict(col_prob=float(A["colp"][i, ch[i]]), min_dist=float(A["md"][i, ch[i]]),
                             max_accel=float(A["acc"][i, ch[i]])))
        fname = f"{names[rec['scene_token']]}_{n}_{kind}_frame{di}.png"
        title = f"{n}  {names[rec['scene_token']]}  frame {di}  cmd {['right','left','straight'][rec['cmd']]}  ({kind})"
        txt = (f"original mode {A['sel'][i]}: col_prob {risk['orig']['col_prob']:.2f} min_dist {risk['orig']['min_dist']:.2f} m acc {risk['orig']['max_accel']:.2f}\n"
               f"plan-2   mode {ch[i]}: col_prob {risk['new']['col_prob']:.2f} min_dist {risk['new']['min_dist']:.2f} m acc {risk['new']['max_accel']:.2f}\n")
        if diag:
            txt += f"GT hit: {diag['gt_class']} at {diag['collision_time_s']:.1f} s (GT size {diag['gt_size'][0]:.2f}x{diag['gt_size'][1]:.2f})\n"
            if diag.get("matched"):
                mt = diag["matched"]
                txt += (f"matched det: cls {mt['cls_score']:.2f} size {mt['size'][0]:.2f}x{mt['size'][1]:.2f} top-mode p {mt['top_mode_prob']:.2f}\n"
                        f"pos err at hit: top {diag['pos_err_top_at_hit']:.2f} m, best mode {diag['pos_err_min_mode_at_hit']:.2f} m\n"
                        f"overlap tests: {' '.join(f'{k}={int(v)}' for k, v in diag['tests'].items())}\n")
            txt += f"cause: {diag['cause']}"
        plans = {"original (rescore + score)": (orig, "#1f77b4"), "plan 2": (new, "#d62728")}
        draw(os.path.join(args.out, fname), title, rec, steps, plans, track, pred, hit[0] if hit else None, None, txt)
        report.append(dict(model=n, kind=kind, frame=di, token=rec["token"], scene=names[rec["scene_token"]],
                           cmd=int(rec["cmd"]), orig_mode=int(A["sel"][i]), new_mode=int(ch[i]), risk=risk,
                           diagnosis=diag, image=fname))
        print(fname, "|", diag["cause"] if diag else "no counted hit")
    with open(os.path.join(args.out, "case_study.json"), "w") as f:
        json.dump(report, f, indent=1)


if __name__ == "__main__":
    main()
