"""Per-decoder-layer losses (0-5) of det_head / map_head from the stage2 training log.

Reports the last logged training value and, for reference, the mean of the last
--tail logged entries (one entry per log_config.interval = 51 iters; batch 4, so a
single entry is noisy).

  python tools/feasibility/layer_losses.py
"""
import argparse
import json
import os

import numpy as np

import common as C

LOG = "work_dirs/distilldrive_stage2_distribution/20261012_201837.log.json"
KEYS = {"det": ["cls", "box", "cns", "yns"], "map": ["cls", "line"]}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--log", default=LOG)
    ap.add_argument("--tail", type=int, default=20)
    args = ap.parse_args()

    path = os.path.join(C.REPO_ROOT, args.log)
    rows = [json.loads(l) for l in open(path) if '"mode": "train"' in l]
    last, tail = rows[-1], rows[-args.tail:]
    out = dict(log=args.log, last_iter=last["iter"], tail_entries=len(tail),
               tail_iters=[tail[0]["iter"], tail[-1]["iter"]], losses={})
    for head, kinds in KEYS.items():
        for kind in kinds:
            name = f"{head}_loss_{kind}"
            out["losses"][name] = dict(
                last=[last[f"{name}_{i}"] for i in range(6)],
                tail_mean=[float(np.mean([r[f"{name}_{i}"] for r in tail])) for i in range(6)],
            )
    C.dump_json(out, os.path.join(C.OUT_ROOT, "stage2_layer_losses.json"))
    print(f"last iter {out['last_iter']}, tail mean over iters {out['tail_iters']}")
    for name, v in out["losses"].items():
        print(f"{name:<16} last " + " ".join(f"{x:.4f}" for x in v["last"])
              + " | tail " + " ".join(f"{x:.4f}" for x in v["tail_mean"]))


if __name__ == "__main__":
    main()
