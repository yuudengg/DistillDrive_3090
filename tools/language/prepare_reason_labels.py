"""Turn a Qwen2.5-VL reason label file (v72 format) into train/val jsonl for train_language.py.  CPU only.

    python tools/language/prepare_reason_labels.py \
        --labels data/language/labels_train.json \
        --pkl data/infos/infos_lang_train.pkl data/infos/infos_lang_val.pkl \
        --out-dir data/language/v72 --val-ratio 0.2

    labelled nuScenes val as a separate file -> it becomes val and --labels is not split:
        --labels data/language/labels_train.json --val-labels data/language/labels_val.json

Input (v72): {sample_token: {command_id, action_lateral, action_longitudinal, reason_lateral, reason_longitudinal,
              map_context, evidence, ...}}
Output rows (what ReasonPacketDataset / ReasonTextDataset read):
    {"sample_token", "meta_action", "meta_action_text", "reason", "scene_token", ...v72 fields kept for analysis}

What it does
  1. meta_action = command_id * 4 + longitudinal  (meta_spec.py order: right 0 / left 1 / straight 2)
     command_id is the same lateral class as gt_ego_fut_cmd, so a lane change counts as left/right like the rule.
     longitudinal: accelerate / start moving -> accelerate, keep speed -> keep, decelerate -> decelerate,
                   stop / stay stopped -> stop
  2. reason = reason_lateral + " " + reason_longitudinal
  3. with --pkl, rows whose 6-step future is cut by the scene end (gt_ego_fut_masks < 1) are dropped:
     the padded (0,0) future makes a moving car look like it stops.
     rows whose reason says the cause is unknown are dropped (written to dropped_unknown.jsonl).
     --keep-unknown keeps them (the LM then learns to answer "unknown").
  4. train/val split BY SCENE (scene_token from the info pkl), so frames of one scene never sit on both sides.
     Val scenes are picked by random search so each meta-action class is spread over train and val as evenly as
     possible and no class is missing from train.
     Without --pkl the scenes are guessed from the file order: the labeller writes samples in info-pkl order
     (scene by scene, in time), and the first frame of a scene has current_speed 0.0 because there is no previous
     frame. A frame with speed 0.0 right after a frame faster than --reset-speed starts a new scene.
     On labels_v72_0_520 this gives 14 blocks of 39-40 frames + 1 frame, i.e. nuScenes-sized scenes.
     Rerun with --pkl when the info pkl is at hand; it is exact.
"""
import argparse
import json
import pickle
import random
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _compat import load_language_package  # noqa: E402

LON_MAP = {
    "accelerate": "accelerate",
    "start moving": "accelerate",
    "keep speed": "keep",
    "decelerate": "decelerate",
    "stop": "stop",
    "stay stopped": "stop",
}
UNKNOWN_RE = re.compile(r"\bunknown\b", re.IGNORECASE)


def convert(token, v, meta_spec, prompt):
    lon = LON_MAP[v["action_longitudinal"]]
    lat = int(v["command_id"])
    meta = lat * meta_spec.NUM_LONGITUDINAL + meta_spec.LONGITUDINAL.index(lon)
    reason = prompt.clean_reason(f"{v['reason_lateral'].strip()} {v['reason_longitudinal'].strip()}")
    row = {"sample_token": token, "meta_action": meta, "meta_action_text": meta_spec.meta_name(meta), "reason": reason}
    row.update({k: v[k] for k in v if k not in row})
    return row


def scene_of(pkls):
    """{token: scene_token}, and the tokens whose 6-step future is cut by the scene end (gt_ego_fut_masks < 1)."""
    out, cut = {}, set()
    for p in pkls:
        with open(p, "rb") as f:
            data = pickle.load(f)
        infos = data["infos"] if isinstance(data, dict) else data
        for info in infos:
            out[info["token"]] = info.get("scene_token")
            masks = info.get("gt_ego_fut_masks")
            if masks is not None and min(float(m) for m in masks) < 1:
                cut.add(info["token"])
    return out, cut


def scenes_from_order(raw, reset_speed):
    """{token: 'order_scene_XX'} from speed resets in file order (see module docstring)."""
    out, sid, prev = {}, 0, None
    for tok, v in raw.items():
        speed = float(v.get("current_speed", 0.0))
        if prev is not None and speed == 0.0 and prev > reset_speed:
            sid += 1
        out[tok] = f"order_scene_{sid:02d}"
        prev = speed
    return out


def split_by_group(rows, groups, val_ratio, seed, tries=5000):
    """Random search over sets of whole groups (scenes) for val.

    score = val size off target + per-class count off target, and a large penalty for every class that
    would be missing from train (a class the model never sees in training cannot be learned).
    """
    by_group = defaultdict(list)
    for r in rows:
        by_group[groups[r["sample_token"]]].append(r)
    keys = sorted(by_group)
    cnt = {g: Counter(r["meta_action"] for r in by_group[g]) for g in keys}
    total = Counter(r["meta_action"] for r in rows)
    n_target = len(rows) * val_ratio

    def score(val_set):
        vc = Counter()
        for g in val_set:
            vc.update(cnt[g])
        s = abs(sum(vc.values()) - n_target)
        s += sum(abs(vc[m] - total[m] * val_ratio) for m in total)
        s += 1000 * sum(1 for m in total if total[m] - vc[m] == 0)
        return s

    rng = random.Random(seed)
    mean = len(rows) / len(keys)
    k0 = max(1, round(n_target / mean))
    best, best_s = None, float("inf")
    for _ in range(tries):
        k = min(len(keys) - 1, max(1, k0 + rng.choice((-1, 0, 0, 0, 1))))
        cand = frozenset(rng.sample(keys, k))
        sc = score(cand)
        if sc < best_s:
            best, best_s = cand, sc
    train = [r for r in rows if groups[r["sample_token"]] not in best]
    val = [r for r in rows if groups[r["sample_token"]] in best]
    return train, val, best


def write_jsonl(path, rows):
    with open(path, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


def hist(rows, meta_spec):
    c = Counter(r["meta_action"] for r in rows)
    return {meta_spec.meta_name(m): c[m] for m in sorted(c)}


def load_rows(paths, keep_unknown, meta_spec, prompt):
    raw = {}
    for p in paths:
        with open(p, encoding="utf-8") as f:
            raw.update(json.load(f))
    rows, dropped, bad = [], [], []
    for tok, v in raw.items():
        row = convert(tok, v, meta_spec, prompt)
        if not keep_unknown and UNKNOWN_RE.search(v["reason_lateral"] + " " + v["reason_longitudinal"]):
            dropped.append(row)
        elif not prompt.validate_reason(row["reason"]):
            bad.append(row)
        else:
            rows.append(row)
    print(f"[labels] {len(raw)} rows -> kept {len(rows)}, unknown dropped {len(dropped)}, invalid {len(bad)}")
    print("[labels] cities:", dict(Counter(v.get("city") for v in raw.values())))
    return raw, rows, dropped, bad


def attach_scenes(rows, scenes, cut):
    n_cut = sum(r["sample_token"] in cut for r in rows)
    if n_cut:  # padded (0,0) future reads as "stop": these action labels are not trustworthy
        print(f"[labels] {n_cut} rows have a cut future (gt_ego_fut_masks < 1) -> dropped")
    rows = [r for r in rows if r["sample_token"] not in cut]
    n_missing = sum(scenes.get(r["sample_token"]) is None for r in rows)
    if n_missing:
        print(f"[warn] {n_missing} tokens not found in the pkl(s); they are left out")
    rows = [r for r in rows if scenes.get(r["sample_token"]) is not None]
    for r in rows:
        r["scene_token"] = scenes[r["sample_token"]]
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--labels", nargs="+", required=True, help="v72 json file(s): {sample_token: {...}}")
    ap.add_argument("--val-labels", nargs="*", default=[],
                    help="separate val label file(s) (e.g. labelled nuScenes val). Given -> no split, these are val")
    ap.add_argument("--pkl", nargs="*", default=[], help="nuScenes info pkl(s) holding these tokens (for scene_token)")
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--val-ratio", type=float, default=0.2)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--keep-unknown", action="store_true")
    ap.add_argument("--reset-speed", type=float, default=0.5, help="no --pkl: speed jump that marks a new scene")
    args = ap.parse_args()
    if args.val_labels and not args.pkl:
        ap.error("--val-labels needs --pkl (to drop cut-future frames and check tokens)")

    lang = load_language_package()
    from importlib import import_module

    meta_spec = import_module(lang.__name__ + ".meta_spec")
    prompt = import_module(lang.__name__ + ".prompt")

    raw, rows, dropped, bad = load_rows(args.labels, args.keep_unknown, meta_spec, prompt)

    if args.pkl:
        scenes, cut = scene_of(args.pkl)
        rows = attach_scenes(rows, scenes, cut)
        groups = {r["sample_token"]: r["scene_token"] for r in rows}
        unit = "scene"
    else:
        print("[warn] no --pkl: scenes guessed from file order + speed resets. Rerun with --pkl to make it exact.")
        guessed = scenes_from_order(raw, args.reset_speed)
        sizes = Counter(guessed.values())
        print("[split] guessed scenes (frames):", [sizes[k] for k in sorted(sizes)])
        for r in rows:
            r["scene_token"] = guessed[r["sample_token"]]
        groups = {r["sample_token"]: r["scene_token"] for r in rows}
        unit = "scene_guessed_from_order"

    if args.val_labels:
        print("[split] val = --val-labels (no split of --labels)")
        _, val, v_dropped, v_bad = load_rows(args.val_labels, args.keep_unknown, meta_spec, prompt)
        val = attach_scenes(val, scenes, cut)
        overlap = {r["scene_token"] for r in rows} & {r["scene_token"] for r in val}
        if overlap:
            print(f"[warn] {len(overlap)} scenes appear in both train and val label files")
        train, val_groups = rows, {r["scene_token"] for r in val}
        dropped, bad = dropped + v_dropped, bad + v_bad
        unit = "separate_val_file"
    else:
        train, val, val_groups = split_by_group(rows, groups, args.val_ratio, args.seed)

    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    write_jsonl(out / "reasons_train.jsonl", train)
    write_jsonl(out / "reasons_val.jsonl", val)
    write_jsonl(out / "dropped_unknown.jsonl", dropped)
    if bad:
        write_jsonl(out / "dropped_invalid.jsonl", bad)
    summary = dict(
        split_unit=unit, scenes_train=len({r["scene_token"] for r in train}), scenes_val=len(val_groups),
        train=len(train), val=len(val), unknown_dropped=len(dropped), invalid=len(bad),
        train_hist=hist(train, meta_spec), val_hist=hist(val, meta_spec), dropped_hist=hist(dropped, meta_spec),
        val_scenes=sorted(val_groups),
    )
    (out / "split_summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False))
    print(json.dumps({k: v for k, v in summary.items() if k != "val_scenes"}, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
