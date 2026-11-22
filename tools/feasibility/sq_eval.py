"""Run queue_matmul_eval.py unchanged and also record, for every frame, the Sparse Query head's
sq_stats (k_eff = agent queries actually kept, n_score_keep = agents forced by det score >= 0.5,
n_gru_source = slots forced so the GRU h0 of ego / rescore rows stays exact).

All arguments are passed through to queue_matmul_eval.py; the statistics go to
<--out-dir>/sq_stats.json (summary + per-frame values).

  PYTHONPATH=. python tools/feasibility/sq_eval.py --variant matmul --no-save \
      --config tools/feasibility/configs/stage2_distribution_det6_map4_sq.py \
      --ckpt work_dirs/recover_stage2_distribution_det6_map4/iter_21096.pth \
      --run sq64_E_seed0_run1 --out-dir work_dirs/sparse_query_eval/E_seed0_q64/run1
"""
import json
import os
import sys

import numpy as np
import torch

import queue_matmul_eval as Q
from projects.mmdet3d_plugin.models.motion.sparse_query_motion_planning_head import SparseQueryMotionPlanningHead

RECORDS = []
_orig_forward = SparseQueryMotionPlanningHead.forward


def _forward(self, *args, **kwargs):
    out = _orig_forward(self, *args, **kwargs)
    if not self.training:
        RECORDS.append({k: int(v) for k, v in self.sq_stats.items() if isinstance(v, (int, np.integer))})
    return out


def main():
    out_dir = sys.argv[sys.argv.index("--out-dir") + 1]
    SparseQueryMotionPlanningHead.forward = _forward
    Q.main()
    keys = sorted({k for r in RECORDS for k in r})
    summary = {k: dict(mean=float(np.mean([r[k] for r in RECORDS if k in r])),
                       min=int(min(r[k] for r in RECORDS if k in r)),
                       max=int(max(r[k] for r in RECORDS if k in r)),
                       n=int(sum(k in r for r in RECORDS))) for k in keys}
    summary["frames"] = len(RECORDS)
    summary["frames_pruned"] = int(sum("k_eff" in r and r["k_eff"] < r.get("n_full", 900) for r in RECORDS))
    with open(os.path.join(out_dir, "sq_stats.json"), "w") as f:
        json.dump(dict(summary=summary, per_frame=RECORDS), f)
    print("sq_stats:", summary)


if __name__ == "__main__":
    torch.multiprocessing.set_start_method("fork")
    main()
