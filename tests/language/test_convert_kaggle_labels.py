"""Kaggle label json -> reasons.jsonl mapping + label checks (tools/language/convert_kaggle_labels.py).

    pytest tests/language -q
"""
import pickle
import sys

from test_language_pipeline import ROOT, lang

sys.path.insert(0, str(ROOT / "tools" / "language"))
from convert_kaggle_labels import convert, diff_actions, missing_from_pkl  # noqa: E402


def row(lat, lon, reason_lon, reason_lat="The route continues straight ahead.", command="straight", **kw):
    return dict(action_lateral=lat, action_longitudinal=lon, reason_lateral=reason_lat, reason_longitudinal=reason_lon,
                command=command, **kw)


def test_convert_and_checks():
    labels = {
        "a": row("change lane left", "stay stopped", "The ego vehicle stays stopped behind a stopped car in its lane.",
                 "The ego vehicle changes into the left lane.", command="turn left"),
        "b": row("straight", "start moving", "The ego vehicle starts moving as the car ahead pulls away."),
        "unk": row("straight", "stop", "The reason is unknown from the visible scene."),
        "you": row("straight", "stop", "You stop because the light in front of you is red."),
        "flip": row("turn left", "keep speed", "The ego vehicle keeps its speed through the turn.", "The ego turns left.",
                    command=0),  # old schema: command id 0 = right
        "ped": row("straight", "accelerate", "The ego vehicle accelerates while a pedestrian is crossing ahead."),
        "miss": {"action_lateral": "straight"},
        "warn": row("straight", "keep speed", "The ego vehicle keeps its speed on a clear road.",
                    map_context={"stop_line_type": "TURN_STOP"},
                    evidence=["pedestrian, 4m ahead in my lane, moving 1m/s", "pedestrian, 4m ahead, leaving my lane"]),
    }
    rows, rejected, warns = convert(labels, lang, label_model="kaggle:test")
    by = {r["sample_token"]: r for r in rows}
    assert lang.meta_name(by["a"]["meta_action"]) == "left+stop"
    assert by["a"]["reason"].startswith("The ego vehicle changes into the left lane. The ego vehicle stays")
    assert lang.meta_name(by["b"]["meta_action"]) == "straight+accelerate"
    assert "route continues" not in by["b"]["reason"]  # straight lateral template is dropped
    flags = {r["sample_token"]: r["flags"] for r in rejected}
    assert flags == {"unk": ["unknown_reason"], "you": ["second_person"], "flip": ["lateral_vs_cmd"],
                     "ped": ["ped_accelerate"], "miss": ["missing_field"]}
    assert warns["warn"] == ["straight_turn_stop", "evidence_conflict"]


def test_diff_and_missing(tmp_path):
    old = {"t1": row("turn right", "stop", "x"), "t2": row("straight", "stop", "x")}
    new = {"t1": row("turn left", "stop", "x"), "t2": row("straight", "stop", "x"), "t3": row("straight", "stop", "x")}
    assert diff_actions(old, new) == [("t1", ("turn right", "stop"), ("turn left", "stop"), True)]

    pkl = tmp_path / "infos.pkl"
    infos = [{"token": t, "scene_token": s} for t, s in [("t1", "s1"), ("t2", "s1"), ("t4", "s1"), ("t9", "s2")]]
    pkl.write_bytes(pickle.dumps({"infos": infos}))
    missing, not_in_pkl = missing_from_pkl(new, [str(pkl)])
    assert missing == {"s1": ["t4"]} and not_in_pkl == 1  # s2 has no labels -> not counted, t3 is not in the pkl
