"""Kaggle label json(s) (labels_vXX_<from>_<to>.json) -> checked reasons.jsonl that train_language.py reads.

    python tools/language/convert_kaggle_labels.py labels_v13_0_260.json labels_v13_110_160.json \
        --out data/language/reasons_kaggle.jsonl \
        --prev labels_v12_0_260.json \
        --pkl data/infos/nuscenes_infos_train.pkl --pkl data/infos/nuscenes_infos_val.pkl \
        --watch c16a 575b 653b e035 48f5 34b9

Outputs
  <out>                  rows that pass every reject check
  <out>.rejected.jsonl   the other rows, with "flags" = why
  <out>.report.md        merge / duplicate / missing / check summary (read this before training)

Mapping to the 12-class meta-action (meta_spec.py):
  * lateral      : turn / change lane left -> left, turn / change lane right -> right, straight -> straight
  * longitudinal : stay stopped -> stop, start moving -> accelerate, the rest 1:1
  * reason       : reason_longitudinal, with reason_lateral in front when the ego is not going straight
                   (the straight lateral reason is one template sentence for ~76% of rows, so it adds nothing)
Several files may overlap in scene range: a later file wins for the same sample token (differences are reported).
train / val are not split here: ReasonTextDataset keeps only the tokens found in the given info pkl.

Checks (REJECT = row left out of training, WARN = report only; the reason sentence is all the LM sees)
  REJECT missing_field     a field needed for the mapping is missing or not a known value
  REJECT invalid_reason    prompt.validate_reason fails (length / banned phrase)
  REJECT unknown_reason    "unknown" in the reason -> would teach the LM to answer "unknown"
  REJECT second_person     you / your in the reason
  REJECT lateral_vs_cmd    action goes left while the route command says right (or the reverse)
  REJECT ped_accelerate    accelerate / start moving while the reason says a pedestrian is crossing
  WARN   straight_turn_stop  command straight but map_context.stop_line_type == TURN_STOP
  WARN   evidence_conflict   same object type both "in my lane" and "leaving my lane" in the evidence
  WARN   reason_off_action   the reason mentions neither part of the action (prompt.action_consistency == 0)
"""
import argparse
import json
import os
import pickle
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _compat import load_language_package  # noqa: E402

LATERAL = {"straight": "straight", "turn left": "left", "change lane left": "left",
           "turn right": "right", "change lane right": "right"}
LONGITUDINAL = {"accelerate": "accelerate", "start moving": "accelerate", "keep speed": "keep",
                "decelerate": "decelerate", "stop": "stop", "stay stopped": "stop"}
COMMAND = {0: "right", 1: "left", 2: "straight", "turn right": "right", "turn left": "left", "straight": "straight"}
REQUIRED = ("action_lateral", "action_longitudinal", "reason_lateral", "reason_longitudinal")


def load_labels(path):
    """json -> {token: row}; keys repeated inside one file are counted (json.load would silently keep the last)"""
    dup = []

    def hook(pairs):
        seen = set()
        for k, _ in pairs:
            if k in seen:
                dup.append(k)
            seen.add(k)
        return dict(pairs)

    with open(path, encoding="utf-8") as f:
        return json.load(f, object_pairs_hook=hook), dup


def build_reason(x):
    parts = [x["reason_longitudinal"]]
    if x["action_lateral"] != "straight":
        parts.insert(0, x["reason_lateral"])
    return " ".join(parts)


def check_row(x, reason, meta, lang):
    """-> (reject flags, warn flags)"""
    rej, warn = [], []
    low = reason.lower()
    if not lang.validate_reason(reason):
        rej.append("invalid_reason")
    if "unknown" in low:
        rej.append("unknown_reason")
    if re.search(r"\byou(r|rs|rself)?\b", low):
        rej.append("second_person")
    cmd = COMMAND.get(x.get("command"))
    if {cmd, LATERAL[x["action_lateral"]]} == {"left", "right"}:
        rej.append("lateral_vs_cmd")
    if LONGITUDINAL[x["action_longitudinal"]] == "accelerate" and re.search(r"pedestrians? (is |are )?crossing", low):
        rej.append("ped_accelerate")
    if cmd == "straight" and (x.get("map_context") or {}).get("stop_line_type") == "TURN_STOP":
        warn.append("straight_turn_stop")
    in_lane, leaving = set(), set()
    for e in x.get("evidence") or []:  # "pedestrian, 4m ahead in my lane, ..." or a sentence "a pedestrian ... is clearing the lane"
        obj = re.search(r"pedestrian|truck|bus|car|cyclist|bicycle|motorcycle|vehicle", e.lower())
        if not obj:
            continue
        if re.search(r"leaving|clearing", e):
            leaving.add(obj.group())
        elif "in my lane" in e:
            in_lane.add(obj.group())
    if in_lane & leaving:
        warn.append("evidence_conflict")
    if lang.action_consistency(reason, meta) == 0:
        warn.append("reason_off_action")
    return rej, warn


def convert(labels, lang, label_model):
    """{token: kaggle row} -> (rows, rejected, warns)   warns = {token: [flags]}"""
    rows, rejected, warns = [], [], {}
    for tok, x in labels.items():
        if any(not isinstance(x.get(k), str) for k in REQUIRED) or x["action_lateral"] not in LATERAL \
                or x["action_longitudinal"] not in LONGITUDINAL:
            rejected.append({"sample_token": tok, "flags": ["missing_field"], "raw": x})
            continue
        meta = lang.meta_index(LATERAL[x["action_lateral"]], LONGITUDINAL[x["action_longitudinal"]])
        reason = lang.clean_reason(build_reason(x))
        row = {"sample_token": tok, "meta_action": meta, "meta_action_text": lang.action_tag(meta),
               "reason": reason, "label_model": label_model}
        rej, warn = check_row(x, reason, meta, lang)
        if warn:
            warns[tok] = warn
        if rej:
            rejected.append(dict(row, flags=rej))
        else:
            rows.append(row)
    return rows, rejected, warns


def action(x):
    return (x.get("action_lateral"), x.get("action_longitudinal"))


def diff_actions(old, new):
    """tokens in both whose action changed -> [(token, old action, new action, lateral flipped?)]"""
    out = []
    for tok in sorted(old.keys() & new.keys()):
        a, b = action(old[tok]), action(new[tok])
        if a != b:
            flip = {LATERAL.get(a[0]), LATERAL.get(b[0])} == {"left", "right"}
            out.append((tok, a, b, flip))
    return out


def missing_from_pkl(labels, pkl_paths):
    """for scenes with >= 1 label: key-frame tokens of that scene that have no label. -> ({scene: [tokens]}, n not in pkl)"""
    by_scene = defaultdict(list)
    for p in pkl_paths:
        with open(p, "rb") as f:
            data = pickle.load(f)
        for info in data["infos"] if isinstance(data, dict) else data:
            by_scene[info["scene_token"]].append(info["token"])
    scene_of = {t: s for s, toks in by_scene.items() for t in toks}
    covered = {scene_of[t] for t in labels if t in scene_of}
    missing = {s: [t for t in by_scene[s] if t not in labels] for s in covered}
    return {s: m for s, m in missing.items() if m}, sum(t not in scene_of for t in labels)


def table(header, rows):
    lines = ["| " + " | ".join(header) + " |", "|" + "---|" * len(header)]
    return lines + ["| " + " | ".join(str(c) for c in r) + " |" for r in rows]


def write_report(path, a):
    L = [f"# label check: {a['label_model']}", ""]
    L += ["## files", *table(["file", "rows", "keys repeated inside the file"], a["files"]), ""]
    L += [f"## merge: {len(a['overlap'])} tokens in 2+ files (later file wins), action differs in {len(a['merge_diff'])}"]
    L += [f"- `{t}` {o} -> {n}" + (" **left/right flip**" if f else "") for t, o, n, f in a["merge_diff"]] + [""]
    if a["prev_diff"] is not None:
        flips = [d for d in a["prev_diff"] if d[3]]
        L += [f"## vs previous version: {len(a['prev_diff'])} actions changed, {len(flips)} left/right flips"]
        L += [f"- `{t}` {o} -> {n}" + (" **left/right flip**" if f else "") for t, o, n, f in a["prev_diff"]] + [""]
    if a["missing"] is not None:
        missing, not_in_pkl = a["missing"]
        L += [f"## missing: {sum(map(len, missing.values()))} unlabeled key frames in {len(missing)} labeled scenes, "
              f"{not_in_pkl} labeled tokens not in the given pkl"]
        L += [f"- scene `{s}`: {len(m)} missing" for s, m in sorted(missing.items(), key=lambda kv: -len(kv[1]))] + [""]
    L += ["## checks", *table(["flag", "rows"], sorted(a["flags"].items(), key=lambda kv: -kv[1])), ""]
    L += ["## unknown reason rate (city x longitudinal action)",
          *table(["city", "action", "unknown / total"], [(c, act, f"{u}/{n}") for (c, act), (u, n) in sorted(a["unknown"].items())]), ""]
    L += ["## kept per meta-action", *table(["meta-action", "rows"], sorted(a["kept"].items())), ""]
    L += ["## watch list"]
    for prefix, hits in a["watch"].items():
        if not hits:
            L.append(f"- `{prefix}`: **not in this batch**")
        for tok, act, flags, reason in hits:
            L.append(f"- `{tok}` {act} flags={flags or '-'}: {reason}")
    with open(path, "w", encoding="utf-8") as f:
        f.write("\n".join(L) + "\n")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("inputs", nargs="+", help="Kaggle label json files, later ones win on duplicate tokens")
    ap.add_argument("--out", required=True)
    ap.add_argument("--prev", nargs="+", default=None, help="previous version's label files, to list changed / flipped actions")
    ap.add_argument("--pkl", action="append", default=None, help="nuScenes info pkl (repeatable) for the missing-frame check")
    ap.add_argument("--watch", nargs="+", default=[], help="token prefixes to show in the report (regression samples)")
    args = ap.parse_args()

    lang = load_language_package()
    labels, files, merge_diff = {}, [], []
    for path in args.inputs:
        cur, dup = load_labels(path)
        files.append((Path(path).name, len(cur), len(dup)))
        merge_diff += diff_actions(labels, cur)
        labels.update(cur)
    overlap = [t for t, n in Counter(t for p in args.inputs for t in load_labels(p)[0]).items() if n > 1]
    prev_diff = None
    if args.prev:
        prev = {}
        for path in args.prev:
            prev.update(load_labels(path)[0])
        prev_diff = diff_actions(prev, labels)

    label_model = "kaggle:" + "+".join(Path(p).stem for p in args.inputs)
    rows, rejected, warns = convert(labels, lang, label_model)
    flags = Counter(f for r in rejected for f in r["flags"]) + Counter(f for w in warns.values() for f in w)
    unknown = defaultdict(lambda: [0, 0])
    for x in labels.values():
        u = unknown[(x.get("city", "?"), x.get("action_longitudinal"))]
        u[0] += "unknown" in (x.get("reason_longitudinal") or "").lower()
        u[1] += 1
    flags_of = {r["sample_token"]: r["flags"] for r in rejected}
    watch = {p: [(t, action(x), flags_of.get(t, []) + warns.get(t, []), x.get("reason_longitudinal"))
                 for t, x in labels.items() if t.startswith(p)] for p in args.watch}

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    for path, items in ((args.out, rows), (args.out + ".rejected.jsonl", rejected)):
        with open(path, "w", encoding="utf-8") as f:
            f.writelines(json.dumps(r, ensure_ascii=False) + "\n" for r in items)
    write_report(args.out + ".report.md", dict(
        label_model=label_model, files=files, overlap=overlap, merge_diff=merge_diff, prev_diff=prev_diff,
        missing=missing_from_pkl(labels, args.pkl) if args.pkl else None, flags=flags, unknown=unknown,
        kept=Counter(lang.meta_name(r["meta_action"]) for r in rows), watch=watch))
    print(f"[done] ok {len(rows)}  rejected {len(rejected)}  warned {len(warns)}  -> {args.out}  (+ .rejected.jsonl, .report.md)")


if __name__ == "__main__":
    main()
