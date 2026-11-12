"""Step 6 (risk score): full-val inference that keeps, per sample, everything the training-free
risk score needs, plus per-sample planning ground-truth metrics.

Nothing in the model, config or checkpoint changes. HierarchicalPlanningDecoder.decode/rescore are
wrapped in this process only, to also return the planning logits *before* the collision-aware
rescore and the rescore's collision mask. Model outputs keep their shapes (planning prediction
[B, 1, 18, 6, 2]); only extra keys are added to the per-sample result dict.

Per sample (list, dataset order):
  idx, token, scene_token, timestamp, map_location, lidar2global [4x4]
  cmd                 0 = right, 1 = left, 2 = straight (one-hot argmax of gt_ego_fut_cmd)
  plan_traj   [18,6,2] all candidates, absolute xy in the lidar frame (= results "planning")
  plan_cls_raw [18]   logits before rescore (row-major: index = cmd * 6 + mode)
  plan_score  [18]    sigmoid of the logits the model uses (cmd row after rescore)
  rescore_col [6]     True where the built-in rescore flagged the cmd-row candidate as colliding
  sel_mode            selected mode within the cmd row; final_planning [6,2]
  agents: boxes [N,9] (x,y,z,w,l,h,yaw,vx,vy), scores [N], cls_scores [N], labels [N],
          trajs [N,6,12,2] absolute, trajs_score [N,6]   (only cls_scores >= --agent-thresh)
  map: vectors [K,20,2], scores [K], labels [K] (0 ped_crossing, 1 divider, 2 boundary)
  gt_traj [6,2], gt_mask [6], gt_fut_boxes: list of 6 arrays [n,7] (planning-eval GT boxes)
  valid               all 6 future steps present (the official planning metric skips others)
  l2 [6], col [6], gt_col [6]   per step, same rules as planning_eval.PlanningMetric

  PYTHONPATH=. python tools/feasibility/dump_planning.py --config ... --ckpt ... --out .../planning_dump.pkl
"""
import argparse
import os
import random
import time

import mmcv
import numpy as np
import torch
from pyquaternion import Quaternion

import common as C
from infer_bench import run_inference

HOLD = {}


def patch_planning_decoder():
    from projects.mmdet3d_plugin.models.motion.decoder import HierarchicalPlanningDecoder as D

    orig_decode, orig_rescore = D.decode, D.rescore

    def rescore(self, plan_cls, *args, **kwargs):
        out = orig_rescore(self, plan_cls, *args, **kwargs)
        HOLD["col"] = ((out - plan_cls) < -500).cpu()
        return out

    def decode(self, det_output, motion_output, planning_output, data):
        HOLD.clear()
        cls = planning_output["classification"][-1]
        raw = cls.reshape(cls.shape[0], -1).float().cpu()  # [B, 18]
        cmd = data["gt_ego_fut_cmd"].argmax(dim=-1).cpu()
        out = orig_decode(self, det_output, motion_output, planning_output, data)
        for i, o in enumerate(out):
            o["planning_cls_raw"] = raw[i]
            o["cmd"] = int(cmd[i])
            o["rescore_col"] = HOLD["col"][i] if "col" in HOLD else torch.zeros(6, dtype=torch.bool)
        return out

    D.decode, D.rescore = decode, rescore


def lidar2global(info):
    l2e = np.eye(4)
    l2e[:3, :3] = Quaternion(info["lidar2ego_rotation"]).rotation_matrix
    l2e[:3, 3] = info["lidar2ego_translation"]
    e2g = np.eye(4)
    e2g[:3, :3] = Quaternion(info["ego2global_rotation"]).rotation_matrix
    e2g[:3, 3] = info["ego2global_translation"]
    return (e2g @ l2e).astype(np.float64)


def compact(res, info, i, agent_thresh):
    r = res["img_bbox"]
    np_ = lambda t: t.numpy() if torch.is_tensor(t) else np.asarray(t)
    keep = np_(r["cls_scores"]) >= agent_thresh
    plan_traj = np_(r["planning"]).reshape(18, 6, 2).astype(np.float32)
    cmd = r["cmd"]
    final = np_(r["final_planning"]).astype(np.float32)
    sel = int(np.argmin(np.abs(plan_traj[cmd * 6:(cmd + 1) * 6] - final).sum(axis=(1, 2))))
    return dict(
        idx=i, token=info["token"], scene_token=info["scene_token"], timestamp=info["timestamp"],
        map_location=info["map_location"], lidar2global=lidar2global(info),
        cmd=cmd, plan_traj=plan_traj,
        plan_cls_raw=np_(r["planning_cls_raw"]).astype(np.float32),
        plan_score=np_(r["planning_score"]).reshape(18).astype(np.float32),
        rescore_col=np_(r["rescore_col"]).astype(bool), sel_mode=sel, final_planning=final,
        agents=dict(
            boxes=np_(r["boxes_3d"])[keep].astype(np.float32),
            scores=np_(r["scores_3d"])[keep].astype(np.float32),
            cls_scores=np_(r["cls_scores"])[keep].astype(np.float32),
            labels=np_(r["labels_3d"])[keep].astype(np.int16),
            trajs=np_(r["trajs_3d"])[keep].astype(np.float16),
            trajs_score=np_(r["trajs_score"])[keep].astype(np.float16),
        ),
        map=dict(
            vectors=np_(r["vectors"]).astype(np.float32) if "vectors" in r else None,
            scores=np_(r["scores"]).astype(np.float32) if "scores" in r else None,
            labels=np_(r["labels"]).astype(np.int16) if "labels" in r else None,
        ),
    )


def add_planning_gt(cfg, records, workers):
    """Per-sample version of planning_eval.planning_eval (same dataset, same metric code)."""
    from mmdet.datasets import build_dataset, build_dataloader
    from projects.mmdet3d_plugin.datasets.evaluation.planning.planning_eval import PlanningMetric

    ds = build_dataset(cfg.eval_config)
    assert [x["token"] for x in ds.data_infos[:len(records)]] == [r["token"] for r in records]
    loader = build_dataloader(ds, samples_per_gpu=1, workers_per_gpu=workers, shuffle=False, dist=False)
    pm = PlanningMetric()
    prog = mmcv.ProgressBar(len(records))
    for i, data in enumerate(loader):
        if i >= len(records):
            break
        rec = records[i]
        gt = data["gt_ego_fut_trajs"].cumsum(dim=-2)[0, :6, :2].float()  # [6, 2]
        mask = data["gt_ego_fut_masks"][0, :6].bool()
        fut_boxes = data["fut_boxes"]
        rec["gt_traj"] = gt.numpy().astype(np.float32)
        rec["gt_mask"] = mask.numpy()
        rec["gt_fut_boxes"] = [fb[0].numpy().astype(np.float32) for fb in fut_boxes]
        rec["valid"] = bool(mask.all())
        assert rec["cmd"] == int(data["gt_ego_fut_cmd"].argmax(dim=-1))
        if rec["valid"]:
            pred = torch.from_numpy(rec["final_planning"][:6]).float()
            rec["l2"] = torch.linalg.norm(pred - gt, dim=-1).numpy()
            rec["gt_col"] = pm.evaluate_single_coll(gt.clone(), fut_boxes).numpy()
            col = pm.evaluate_single_coll(pred.clone(), fut_boxes)
            rec["col"] = (col & ~torch.from_numpy(rec["gt_col"])).numpy()
        prog.update()


def official_style(records):
    """Repo planning metric (cumulative mean over steps, avg of 1/2/3 s) from per-sample values."""
    v = [r for r in records if r["valid"]]
    out = {}
    for k in ["l2", "col", "gt_col"]:
        per_t = np.stack([r[k] for r in v]).astype(np.float64).mean(0)
        cum = np.cumsum(per_t) / np.arange(1, 7)
        out[k] = float(cum[[1, 3, 5]].mean())
    out["n_valid"] = len(v)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=C.DEFAULT_CONFIG)
    ap.add_argument("--ckpt", default=C.DEFAULT_CKPT)
    ap.add_argument("--out", required=True, help="output pkl path")
    ap.add_argument("--variant", choices=["original", "matmul"], default="matmul")
    ap.add_argument("--agent-thresh", type=float, default=0.05)
    ap.add_argument("--limit", type=int, default=None, help="first N frames only (smoke test)")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--workers", type=int, default=4)
    args = ap.parse_args()

    cfg = C.load_cfg(args.config)
    random.seed(args.seed); np.random.seed(args.seed)
    torch.manual_seed(args.seed); torch.cuda.manual_seed_all(args.seed)
    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)

    model, _ = C.build_model(cfg, ckpt=args.ckpt)
    C.apply_precision(model, "repo")
    model.cuda().eval()
    C.set_queue_variant(args.variant)
    patch_planning_decoder()

    dataset, loader = C.build_test_loader(cfg)
    t0 = time.time()
    results = run_inference(model, loader, args.limit)
    t_inf = (time.time() - t0) / 60
    records = [compact(r, dataset.data_infos[i], i, args.agent_thresh) for i, r in enumerate(results)]
    del results
    add_planning_gt(cfg, records, args.workers)
    meta = dict(config=args.config, ckpt=args.ckpt, variant=args.variant, mode="repo",
                seed=args.seed, agent_thresh=args.agent_thresh, inference_min=t_inf,
                frame="lidar (x right, y forward), metres; planning steps 0.5 s",
                n_records=len(records), metric=official_style(records))
    mmcv.dump(dict(meta=meta, records=records), args.out)
    print(meta)


if __name__ == "__main__":
    torch.multiprocessing.set_start_method("fork")
    main()
