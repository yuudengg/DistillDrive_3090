"""Raw per-frame dump of the final model E for the language module (LangPacketBuilder input is built
from this later; no format conversion here). Train and val, model outputs + plan 2 + risk metrics.

Nothing in the model, config or checkpoint changes. Inference = the E evaluation setting
(tools/feasibility/queue_matmul_eval.py --variant source --mode repo, seed 0): repo precision,
InstanceQueue.prepare_motion of the source (matmul since a567809), test pipeline (no augmentation),
batch 1, shuffle=False, dataset order = infos sorted by timestamp (scenes stay contiguous).
HierarchicalPlanningDecoder.decode/rescore and InstanceBank.get are wrapped in this process only to
also return the raw planning deltas, the pre-rescore logits, the rescore collision mask and whether
the temporal memory was carried over from the previous frame. Output shapes are not changed.

Stages
  infer   GPU. Streams inference and writes one raw file per scene: <out>/_raw/<split>/<scene>.pkl
  post    CPU. Plan 2 (reselect_v3.choose "v2", detection score >= 0.1, tau = 0.25 m, the risk-score
          branch code as is), pred risk metrics (risk_eval.pred_family, agents with cls >= 0.1),
          analysis-only GT (planning-eval dataset, PlanningMetric.evaluate_single_coll), then the
          final <out>/<split>/<scene>.pkl and <out>/index.json
  verify  CPU. Frame counts, L2 / obj_box_col (A and B), trigger / change rates, and 3 random frames
          against the outputs.pkl of an official queue_matmul_eval.py run

  PYTHONPATH=. python tools/feasibility/lang_dump.py infer --split val [--max-scenes 5] --out work_dirs/lang_dump
  PYTHONPATH=. python tools/feasibility/lang_dump.py post  --split val --out work_dirs/lang_dump
  PYTHONPATH=. python tools/feasibility/lang_dump.py verify --out work_dirs/lang_dump --official .../outputs.pkl
"""
import argparse
import glob
import json
import os
import pickle
import random
import time
from multiprocessing import Pool

import numpy as np
import torch

import common as C

CONFIG = "tools/feasibility/configs/stage2_distribution_det6_map4.py"
CKPT = "work_dirs/recover_stage2_distribution_det6_map4/iter_21096.pth"
INFOS = {"train": "data/infos/nuscenes_infos_train.pkl", "val": "data/infos/nuscenes_infos_val.pkl"}
PLAN2 = dict(variant="v2", det_thresh=0.1, tau=0.25)        # reselect_trainset.json "chosen" (thr 0.1, tau 0.25)
RISK_DET_THRESH = 0.1                                         # same agents as plan 2
RISK_KEYS = ["min_dist", "col_prob", "ttc", "drac", "max_accel", "max_jerk"]
HOLD = {}


# ------------------------------------------------------------------------------------- infer
def patch_model():
    from projects.mmdet3d_plugin.models.motion.decoder import HierarchicalPlanningDecoder as D
    from projects.mmdet3d_plugin.models.instance_bank import InstanceBank as IB

    orig_decode, orig_rescore, orig_get = D.decode, D.rescore, IB.get

    def rescore(self, plan_cls, *a, **k):
        out = orig_rescore(self, plan_cls, *a, **k)
        HOLD["col"] = ((out - plan_cls) < -500).cpu()
        return out

    def decode(self, det_output, motion_output, planning_output, data):
        HOLD.pop("col", None)
        cls = planning_output["classification"][-1]
        raw = cls.reshape(cls.shape[0], -1).float().cpu()                       # [B, 18] logits
        delta = planning_output["prediction"][-1].reshape(cls.shape[0], -1, self.ego_fut_ts, 2).float().cpu()
        cmd = data["gt_ego_fut_cmd"].argmax(dim=-1).cpu()
        out = orig_decode(self, det_output, motion_output, planning_output, data)
        for i, o in enumerate(out):
            o["planning_cls_raw"] = raw[i]
            o["planning_delta"] = delta[i]                                      # [18, 6, 2] before cumsum
            o["cmd"] = int(cmd[i])
            o["rescore_col"] = HOLD["col"][i] if "col" in HOLD else torch.zeros(6, dtype=torch.bool)
        return out

    def get(self, batch_size, metas=None, dn_metas=None):
        out = orig_get(self, batch_size, metas, dn_metas)
        if getattr(self, "_lang_dump_det", False):
            HOLD["mem"] = self.mask is not None and bool(self.mask.reshape(-1)[0])
        return out

    D.decode, D.rescore, IB.get = decode, rescore, get


def lidar2global(info):
    from pyquaternion import Quaternion
    l2e = np.eye(4)
    l2e[:3, :3] = Quaternion(info["lidar2ego_rotation"]).rotation_matrix
    l2e[:3, 3] = info["lidar2ego_translation"]
    e2g = np.eye(4)
    e2g[:3, :3] = Quaternion(info["ego2global_rotation"]).rotation_matrix
    e2g[:3, 3] = info["ego2global_translation"]
    return e2g @ l2e


def to_np(t, dt=None):
    a = t.detach().cpu().numpy() if torch.is_tensor(t) else np.asarray(t)
    return a.astype(dt) if dt is not None else a


def raw_record(res, info, idx, mem_kept):
    r = res["img_bbox"]
    cmd = r["cmd"]
    plan_traj = to_np(r["planning"], np.float32).reshape(18, 6, 2)
    delta = to_np(r["planning_delta"], np.float32).reshape(18, 6, 2)
    final = to_np(r["final_planning"], np.float32)
    sel = int(np.argmin(np.abs(plan_traj[cmd * 6:(cmd + 1) * 6] - final).sum(axis=(1, 2))))
    assert np.array_equal(plan_traj[cmd * 6 + sel], final)
    ego = np.asarray(info["ego_status"], np.float32)
    det = dict(
        boxes=to_np(r["boxes_3d"], np.float32), scores=to_np(r["scores_3d"], np.float32),
        cls_scores=to_np(r["cls_scores"], np.float32), labels=to_np(r["labels_3d"], np.int16),
        instance_ids=to_np(r["instance_ids"], np.int64) if "instance_ids" in r else None,
        trajs=to_np(r["trajs_3d"], np.float32), trajs_score=to_np(r["trajs_score"], np.float32))
    det["trajs_prob"] = det["trajs_score"] / np.maximum(det["trajs_score"].sum(-1, keepdims=True), 1e-9)
    return dict(
        idx=idx, sample_token=info["token"], scene_token=info["scene_token"], timestamp=int(info["timestamp"]),
        map_location=info["map_location"], lidar2global=lidar2global(info),
        cmd=cmd, cmd_onehot=np.asarray(info["gt_ego_fut_cmd"], np.float32),
        memory_carried=bool(mem_kept),
        det=det,
        map=dict(vectors=to_np(r["vectors"], np.float32), scores=to_np(r["scores"], np.float32),
                 labels=to_np(r["labels"], np.int16)),
        plan=dict(delta=delta, traj=plan_traj, cls_raw=to_np(r["planning_cls_raw"], np.float32),
                  score=to_np(r["planning_score"], np.float32).reshape(18),
                  rescore_col=to_np(r["rescore_col"]).astype(bool)),
        A=dict(mode=sel, index=cmd * 6 + sel, traj=final),
        ego=dict(ego_status=ego, speed=float(ego[6]), can_bus_available=bool(np.any(ego != 0))),
    )


def stage_infer(args):
    cfg = C.load_cfg(CONFIG)
    ann = INFOS[args.split]
    cfg.data.test.ann_file = ann
    cfg.eval_config.ann_file = ann
    if "eval_config" in cfg.data.test:
        cfg.data.test.eval_config.ann_file = ann
    random.seed(0); np.random.seed(0); torch.manual_seed(0); torch.cuda.manual_seed_all(0)
    model, _ = C.build_model(cfg, ckpt=CKPT)
    C.apply_precision(model, "repo")
    model.cuda().eval()
    C.set_queue_variant("source")
    patch_model()
    model.head.det_head.instance_bank._lang_dump_det = True
    dataset, loader = C.build_test_loader(cfg, workers=args.workers)
    infos = dataset.data_infos
    raw_dir = os.path.join(args.out, "_raw", args.split)
    os.makedirs(raw_dir, exist_ok=True)

    import mmcv
    t0 = time.time()
    buf, scenes_done, n = [], 0, 0
    total = len(infos) if args.max_scenes is None else None
    prog = mmcv.ProgressBar(total or 1)

    def flush():
        with open(os.path.join(raw_dir, buf[0]["scene_token"] + ".pkl"), "wb") as f:
            pickle.dump(buf, f, protocol=4)

    with torch.no_grad():
        for i, data in enumerate(loader):
            info = infos[i]
            if buf and info["scene_token"] != buf[-1]["scene_token"]:
                flush(); buf = []; scenes_done += 1
                if args.max_scenes is not None and scenes_done >= args.max_scenes:
                    break
            HOLD.pop("mem", None)
            res = model(return_loss=False, rescale=True, **C.to_gpu(data))
            assert len(res) == 1
            buf.append(raw_record(res[0], info, i, HOLD["mem"]))
            n += 1
            if total:
                prog.update()
        else:
            flush(); scenes_done += 1
    meta = dict(split=args.split, frames=n, scenes=scenes_done, inference_min=(time.time() - t0) / 60,
                config=CONFIG, ckpt=CKPT, mode="repo", queue_variant="source", seed=0, ann_file=ann)
    with open(os.path.join(raw_dir, "_meta.json"), "w") as f:
        json.dump(meta, f, indent=1)
    print("\n", meta)


# ------------------------------------------------------------------------------------- post
G = {}


def risk_rec(rec):
    """The dict layout risk_eval.pred_family / reselect_v2.online_scores read."""
    d = rec["det"]
    return dict(plan_traj=rec["plan"]["traj"], plan_cls_raw=rec["plan"]["cls_raw"], cmd=rec["cmd"],
                agents=dict(boxes=d["boxes"], cls_scores=d["cls_scores"], labels=d["labels"],
                            trajs=d["trajs"], trajs_score=d["trajs_score"]),
                map=rec["map"])


def post_scene(path):
    import reselect_v3 as V3
    from reselect_v2 import online_scores
    from risk_eval import pred_family
    with open(path, "rb") as f:
        recs = pickle.load(f)
    td = np.arange(0, 3 + 1e-9, 0.1)
    rargs = argparse.Namespace(agent_thresh=RISK_DET_THRESH, map_thresh=0.3, dt=0.1, drac_min_gap=0.1, drac_cap=20.0)
    out, maxdiff = {}, 0.0
    for rec in recs:
        rr = risk_rec(rec)
        c, v0 = rec["cmd"], rec["ego"]["speed"]
        rows = list(range(c * 6, c * 6 + 6))
        pf = pred_family(rr, v0, rargs, td)
        risk6 = {k: pf[f"pred_{k}"][rows].astype(np.float32) for k in RISK_KEYS}
        colp, md, acc = online_scores(rr, rows, v0, td, agent_thresh=PLAN2["det_thresh"], prune=True)
        maxdiff = max(maxdiff, float(np.abs(colp - risk6["col_prob"]).max()),
                      float(np.abs(np.minimum(md, 1e6) - np.minimum(risk6["min_dist"], 1e6)).max()),
                      float(np.abs(acc - risk6["max_accel"]).max()))
        A = dict(sel=np.array([rec["A"]["mode"]]), colp=colp[None], md=md[None], acc=risk6["max_accel"][None].astype(np.float64))
        ch, trig = V3.choose(A, np.ones(1, bool), PLAN2["tau"], PLAN2["variant"])
        mb = int(ch[0])
        out[rec["sample_token"]] = dict(
            B=dict(triggered=bool(trig[0]), mode=mb, index=c * 6 + mb, traj=rec["plan"]["traj"][c * 6 + mb].copy(),
                   changed=bool(mb != rec["A"]["mode"])),
            risk=dict(cand=risk6, A={k: float(v[rec["A"]["mode"]]) for k, v in risk6.items()},
                      B={k: float(v[mb]) for k, v in risk6.items()}))
    return path, out, maxdiff


def gt_pass(split, sel_trajs, workers):
    """Per-sample planning GT and A/B metrics with the planning-eval dataset (PlanningMetric rules)."""
    import mmcv
    from mmdet.datasets import build_dataset, build_dataloader
    from projects.mmdet3d_plugin.datasets.evaluation.planning.planning_eval import PlanningMetric
    cfg = C.load_cfg(CONFIG)
    cfg.eval_config.ann_file = INFOS[split]
    ds = build_dataset(cfg.eval_config)
    loader = build_dataloader(ds, samples_per_gpu=1, workers_per_gpu=workers, shuffle=False, dist=False)
    pm = PlanningMetric()
    out = {}
    prog = mmcv.ProgressBar(len(ds))
    for i, data in enumerate(loader):
        tok = ds.data_infos[i]["token"]
        prog.update()
        if tok not in sel_trajs:
            continue
        cmd, ta, tb = sel_trajs[tok]
        gt = data["gt_ego_fut_trajs"].cumsum(dim=-2)[0, :6, :2].float()
        mask = data["gt_ego_fut_masks"][0, :6].bool()
        assert cmd == int(data["gt_ego_fut_cmd"].argmax(dim=-1))
        fb = data["fut_boxes"]
        o = dict(gt_traj=gt.numpy().astype(np.float32), gt_mask=mask.numpy(), valid=bool(mask.all()))
        if o["valid"]:
            gcol = pm.evaluate_single_coll(gt.clone(), fb)
            o["gt_col"] = gcol.numpy()
            for name, t in (("A", ta), ("B", tb)):
                p = torch.from_numpy(t[:6]).float()
                o[f"{name}_l2"] = torch.linalg.norm(p - gt, dim=-1).numpy().astype(np.float32)
                o[f"{name}_col"] = (pm.evaluate_single_coll(p.clone(), fb) & ~gcol).numpy()
        out[tok] = o
    return out


def stage_post(args):
    raw_dir = os.path.join(args.out, "_raw", args.split)
    paths = sorted(glob.glob(os.path.join(raw_dir, "*.pkl")))
    t0 = time.time()
    with Pool(args.cpu) as pool:
        res = pool.map(post_scene, paths, chunksize=1)
    maxdiff = max(r[2] for r in res)
    print(f"plan2/risk done ({(time.time() - t0) / 60:.1f} min), online_scores vs pred_family max |diff| {maxdiff:.2e}")
    plan2 = {tok: v for _, o, _ in res for tok, v in o.items()}
    sel = {}
    for p in paths:
        with open(p, "rb") as f:
            for rec in pickle.load(f):
                sel[rec["sample_token"]] = (rec["cmd"], rec["A"]["traj"], plan2[rec["sample_token"]]["B"]["traj"])
    t1 = time.time()
    gt = gt_pass(args.split, sel, args.workers)
    print(f"\nGT pass done ({(time.time() - t1) / 60:.1f} min)")
    out_dir = os.path.join(args.out, args.split)
    os.makedirs(out_dir, exist_ok=True)
    index = {}
    for p in paths:
        with open(p, "rb") as f:
            recs = pickle.load(f)
        for rec in recs:
            tok = rec["sample_token"]
            rec.update(plan2[tok])
            rec["analysis_gt_only"] = gt[tok]
        scene = recs[0]["scene_token"]
        rel = os.path.join(args.split, scene + ".pkl")
        with open(os.path.join(args.out, rel), "wb") as f:
            pickle.dump(recs, f, protocol=4)
        for k, rec in enumerate(recs):
            index[rec["sample_token"]] = dict(file=rel, pos=k)
    with open(os.path.join(raw_dir, "_meta.json")) as f:
        meta = json.load(f)
    meta.update(post_min=(time.time() - t0) / 60, plan2_vs_risk_max_abs_diff=maxdiff)
    with open(os.path.join(out_dir, "_meta.json"), "w") as f:
        json.dump(meta, f, indent=1)
    write_index(args.out, args.split, index)
    if not args.keep_raw:
        for p in paths:
            os.remove(p)
        os.remove(os.path.join(raw_dir, "_meta.json"))
        os.rmdir(raw_dir)
    print("post done:", meta)


FIELDS = {
    "_conventions": {
        "frame": "lidar frame of the current sample: x right, y forward, z up, metres, origin = lidar (the "
                 "planning / evaluation frame of the repo). lidar2global [4,4] maps it to nuScenes global.",
        "time": "planning / motion steps every 0.5 s (step k = (k+1)*0.5 s); timestamp in microseconds",
        "candidate index": "index = cmd * 6 + mode, cmd 0 = turn right, 1 = turn left, 2 = go straight",
        "split order": "frames inside a scene file are in time order (dataset order, infos sorted by timestamp)",
    },
    "idx": ("int", "", "position in the split's dataset order"),
    "sample_token / scene_token": ("str", "", "nuScenes tokens"),
    "timestamp": ("int", "us", "sample timestamp"),
    "map_location": ("str", "", "nuScenes map name"),
    "lidar2global": ("[4,4] float64", "m", "current lidar frame -> global"),
    "cmd": ("int", "", "driving command 0 right / 1 left / 2 straight (model input, argmax of cmd_onehot)"),
    "cmd_onehot": ("[3] float32", "", "gt_ego_fut_cmd as fed to the model"),
    "memory_carried": ("bool", "", "temporal memory (instance bank) was carried over from the previous frame; False = "
                       "reset. The model resets only when |dt| > 2 s, so at a scene change inside the same log "
                       "the memory is carried (same as the official evaluation)"),
    "det.boxes": ("[300,10] float32", "m, rad, m/s", "x, y, z, w, l, h, yaw, vx, vy, vz (lidar frame); all 300 "
                  "decoder outputs, no score threshold, sorted by scores"),
    "det.scores": ("[300] float32", "", "cls score x centerness (used by the nuScenes detection eval)"),
    "det.cls_scores": ("[300] float32", "", "class confidence (plan 2 / risk metrics keep cls_scores >= 0.1)"),
    "det.labels": ("[300] int16", "", "0 car, 1 truck, 2 construction_vehicle, 3 bus, 4 trailer, 5 barrier, "
                   "6 motorcycle, 7 bicycle, 8 pedestrian, 9 traffic_cone"),
    "det.instance_ids": ("[300] int64", "", "tracking id from the instance bank (stable across frames while tracked)"),
    "det.trajs": ("[300,6,12,2] float32", "m", "predicted future positions per mode, absolute xy, 12 steps x 0.5 s"),
    "det.trajs_score": ("[300,6] float32", "", "mode score (sigmoid), as output"),
    "det.trajs_prob": ("[300,6] float32", "", "trajs_score normalised to sum 1 per agent (what the risk metrics use)"),
    "map.vectors": ("[100,20,2] float32", "m", "map element polylines, 20 points, xy lidar frame"),
    "map.scores": ("[100] float32", "", "confidence"),
    "map.labels": ("[100] int16", "", "0 ped_crossing, 1 divider, 2 boundary"),
    "plan.delta": ("[18,6,2] float32", "m", "raw model output: displacement of each 0.5 s step"),
    "plan.traj": ("[18,6,2] float32", "m", "cumulative positions = cumsum(delta) (what the model's decode returns)"),
    "plan.cls_raw": ("[18] float32", "logit", "candidate scores before the collision-aware rescore"),
    "plan.score": ("[18] float32", "", "sigmoid of the scores used for selection (cmd row after rescore, others raw)"),
    "plan.rescore_col": ("[6] bool", "", "built-in rescore flagged the cmd-row candidate as colliding (-999)"),
    "A": ("dict", "", "before plan 2: mode (0..5 in the cmd row), index (0..17), traj [6,2] m = model's final_planning"),
    "B": ("dict", "", "after plan 2 (v2, det cls >= 0.1, tau 0.25 m, accel filter +1.0 m/s^2): triggered, mode, index, "
          "traj [6,2] m, changed (mode != A.mode)"),
    "risk.cand": ("dict of [6] float32", "", "pred risk metrics of the 6 cmd-row candidates (mode order): min_dist m, "
                  "col_prob 0..1, ttc s (inf = none within 3 s), drac m/s^2 (cap 20), max_accel m/s^2, max_jerk m/s^3; "
                  "agents with cls >= 0.1, most likely mode, 0.1 s grid over 3 s"),
    "risk.A / risk.B": ("dict of float", "", "the same metrics for the selected candidate A / B"),
    "ego.ego_status": ("[10] float32", "", "CAN bus: accel xyz m/s^2, rotation rate xyz rad/s, velocity xyz m/s (ego frame), "
                       "steering (rad, + left); all 0 when the scene has no CAN bus"),
    "ego.speed": ("float", "m/s", "ego_status[6], forward speed (plan 2 / risk metric v0)"),
    "ego.can_bus_available": ("bool", "", "False when ego_status is all zero (no CAN bus for the scene)"),
    "analysis_gt_only": ("dict", "", "GROUND TRUTH, NOT A MODEL INPUT - for later comparison only: gt_traj [6,2] m "
                         "(cumulative), gt_mask [6], valid (all 6 steps), and when valid gt_col [6], A_l2 / B_l2 [6] m, "
                         "A_col / B_col [6] (obj_box_col rule of planning_eval)"),
}


def write_index(out, split, index):
    path = os.path.join(out, "index.json")
    d = {"fields": FIELDS, "samples": {}, "splits": {}}
    if os.path.exists(path):
        with open(path) as f:
            d = json.load(f)
    d["fields"] = FIELDS
    d["samples"] = {k: v for k, v in d["samples"].items() if not v["file"].startswith(split + "/")}
    d["samples"].update(index)
    d["splits"][split] = len(index)
    d["plan2"], d["config"], d["ckpt"] = PLAN2, CONFIG, CKPT
    with open(path, "w") as f:
        json.dump(d, f, indent=1)


# ------------------------------------------------------------------------------------- verify
def load_split(out, split):
    recs = []
    for p in sorted(glob.glob(os.path.join(out, split, "*.pkl"))):
        with open(p, "rb") as f:
            recs.extend(pickle.load(f))
    return sorted(recs, key=lambda r: r["idx"])


def official_style(per_t):
    m = np.stack(per_t).astype(np.float64).mean(0)
    cum = np.cumsum(m) / np.arange(1, 7)
    return float(cum[[1, 3, 5]].mean())


def stage_verify(args):
    res = {}
    for split in ["train", "val"]:
        recs = load_split(args.out, split)
        if not recs:
            continue
        v = [r for r in recs if r["analysis_gt_only"]["valid"]]
        g = lambda k: [r["analysis_gt_only"][k] for r in v]
        res[split] = dict(
            frames=len(recs), scenes=len({r["scene_token"] for r in recs}), valid=len(v),
            idx_contiguous=[r["idx"] for r in recs] == list(range(len(recs))),
            A_L2=official_style(g("A_l2")), A_obj_box_col=official_style(g("A_col")),
            B_L2=official_style(g("B_l2")), B_obj_box_col=official_style(g("B_col")),
            trigger_rate_valid=float(np.mean([r["B"]["triggered"] for r in v])),
            changed_rate_valid=float(np.mean([r["B"]["changed"] for r in v])),
            trigger_rate_all=float(np.mean([r["B"]["triggered"] for r in recs])),
            changed_rate_all=float(np.mean([r["B"]["changed"] for r in recs])),
            memory_reset_frames=int(sum(not r["memory_carried"] for r in recs)),
            can_bus_missing_frames=int(sum(not r["ego"]["can_bus_available"] for r in recs)),
            delta_cumsum_max_abs_diff=float(max(np.abs(np.cumsum(r["plan"]["delta"], 1) - r["plan"]["traj"]).max() for r in recs)),
        )
        # memory flag vs the timestamp rule of InstanceBank (|dt| <= 2 s with a cached previous frame)
        ts = np.array([r["timestamp"] for r in recs]) / 1e6
        rule = np.concatenate([[False], np.abs(np.diff(ts)) <= 2])
        res[split]["memory_flag_matches_dt_rule"] = bool((rule == np.array([r["memory_carried"] for r in recs])).all())
        sc = [r["scene_token"] for r in recs]
        res[split]["scene_changes_with_memory_carried"] = int(sum(sc[k] != sc[k - 1] and recs[k]["memory_carried"]
                                                               for k in range(1, len(recs))))
        print(split, res[split], flush=True)
    if args.official:
        import mmcv
        off = mmcv.load(args.official)
        recs = load_split(args.out, "val")
        assert len(off) == len(recs)
        rng = np.random.RandomState(args.seed)
        picks = sorted(rng.choice(len(recs), 3, replace=False).tolist())
        cmp = []
        for i in picks:
            o, r = off[i]["img_bbox"], recs[i]
            ft = o["final_planning"].numpy()
            pt = o["planning"].numpy().reshape(18, 6, 2)
            cmp.append(dict(idx=i, sample_token=r["sample_token"],
                            final_planning_max_abs_diff=float(np.abs(ft - r["A"]["traj"]).max()),
                            all18_max_abs_diff=float(np.abs(pt - r["plan"]["traj"]).max()),
                            same_selected_index=bool(np.argmin(np.abs(pt[r["cmd"] * 6:r["cmd"] * 6 + 6] - ft).sum((1, 2))) == r["A"]["mode"])))
        res["official_compare"] = cmp
        print(cmp)
    with open(os.path.join(args.out, "verify.json"), "w") as f:
        json.dump(res, f, indent=1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("stage", choices=["infer", "post", "verify"])
    ap.add_argument("--split", choices=["train", "val"])
    ap.add_argument("--out", default="work_dirs/lang_dump")
    ap.add_argument("--max-scenes", type=int, default=None, help="infer: stop after N scenes (size estimate)")
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--cpu", type=int, default=12)
    ap.add_argument("--keep-raw", action="store_true")
    ap.add_argument("--official", default=None, help="verify: outputs.pkl of queue_matmul_eval.py on val")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    args.out = os.path.abspath(args.out)
    {"infer": stage_infer, "post": stage_post, "verify": stage_verify}[args.stage](args)


if __name__ == "__main__":
    torch.multiprocessing.set_start_method("fork")
    main()
