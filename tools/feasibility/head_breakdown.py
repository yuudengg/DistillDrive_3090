"""Step 3 (Exp 5 prep): det_head / map_head split per decoder layer and per op.

Categories inside each decoder layer (a layer ends at its `refine` op):
  self_attn / temporal_attn : Sparse4DHead.graph_model (pos-embedding cat, fc_before,
                              MultiheadFlashAttention -> flash-attn kernel, fc_after)
  deformable                : DeformableFeatureAggregation (incl. the custom CUDA
                              deformable_aggregation op)
  ffn                       : AsymmetricFFN; nested inside it
                                ffn.hidden   = layers[0]: Linear(in, hidden) + ReLU + Dropout
                                ffn.out_proj = layers[1]: Linear(hidden, embed)
  refine                    : refinement module (box / cls / quality branches)
  anchor_enc                : anchor_encoder (2 calls before the layers = "pre",
                              then one after each refine)
  norm                      : LayerNorm ops of operation_order
  other                     : head total - all of the above (instance bank get /
                              update / cache, instance id, output assembly)

Training memory : train_memory.py --hooks heads (this module's wrappers + SegmentMemory)
Inference time  : this script, CUDA events around every segment (no extra syncs),
                  deploy model, repo mode, matmul queue, batch 1, 10 warmup + 100 frames.

  PYTHONPATH=. python tools/feasibility/head_breakdown.py --infer
  PYTHONPATH=. python tools/feasibility/head_breakdown.py --summarize
"""
import argparse
import json
import os
from collections import defaultdict

import numpy as np
import torch

import common as C

HEADS = (("det", "det_head"), ("map", "map_head"))
CAT = {"temp_gnn": "temporal_attn", "gnn": "self_attn", "deformable": "deformable",
       "ffn": "ffn", "refine": "refine", "norm": "norm"}
CATEGORIES = ["self_attn", "temporal_attn", "deformable", "ffn", "refine", "anchor_enc", "norm", "other"]


def layer_of_ops(operation_order):
    """decoder layer index of every op (a layer ends at refine)."""
    out, k = [], 0
    for op in operation_order:
        out.append(k)
        if op == "refine":
            k += 1
    return out


def install_head_wrappers(model, rec):
    """rec: object with pre(name) / post(name). Names: '<det|map>.<L#|pre>.<category>[.sub]'
    and '<det|map>.total'."""
    for prefix, attr in HEADS:
        head = getattr(model.head, attr)
        layer = layer_of_ops(head.operation_order)
        state = dict(active=False, cur="pre")

        def wrap(fn, name_fn, _s=state):
            def wrapped(*a, **k):
                if not _s["active"]:
                    return fn(*a, **k)
                name = name_fn(*a, **k)
                rec.pre(name)
                out = fn(*a, **k)
                rec.post(name)
                return out
            return wrapped

        total_fwd = head.forward

        def total(*a, _f=total_fwd, _p=prefix, _s=state, **k):
            _s["active"], _s["cur"] = True, "pre"
            rec.pre(f"{_p}.total")
            out = _f(*a, **k)
            rec.post(f"{_p}.total")
            _s["active"] = False
            return out

        head.forward = total
        head.graph_model = wrap(
            head.graph_model,
            lambda index, *a, _p=prefix, _l=layer, _o=head.operation_order, **k:
                f"{_p}.L{_l[index]}.{CAT[_o[index]]}")
        for i, op in enumerate(head.operation_order):
            m = head.layers[i]
            if m is None or op not in ("norm", "ffn", "deformable", "refine"):
                continue
            name = f"{prefix}.L{layer[i]}.{CAT[op]}"
            if op == "refine":
                def after_refine(*a, _f=m.forward, _n=name, _s=state, _k=layer[i], **k):
                    if not _s["active"]:
                        return _f(*a, **k)
                    rec.pre(_n)
                    out = _f(*a, **k)
                    rec.post(_n)
                    _s["cur"] = _k
                    return out
                m.forward = after_refine
            else:
                m.forward = wrap(m.forward, lambda *a, _n=name, **k: _n)
            if op == "ffn":
                m.layers[0].forward = wrap(m.layers[0].forward, lambda *a, _n=name, **k: _n + ".hidden")
                m.layers[1].forward = wrap(m.layers[1].forward, lambda *a, _n=name, **k: _n + ".out_proj")
        head.anchor_encoder.forward = wrap(
            head.anchor_encoder.forward,
            lambda *a, _p=prefix, _s=state, **k:
                f"{_p}.{'pre' if _s['cur'] == 'pre' else 'L%d' % _s['cur']}.anchor_enc")


class EventTimer:
    """CUDA-event timing of (possibly nested) segments; one sync per frame."""

    def __init__(self):
        self._open = {}
        self._pending = []
        self.frames = []

    def pre(self, name):
        e = torch.cuda.Event(enable_timing=True)
        e.record()
        self._open[name] = e

    def post(self, name):
        e = torch.cuda.Event(enable_timing=True)
        e.record()
        self._pending.append((name, self._open.pop(name), e))

    def end_frame(self, keep):
        torch.cuda.synchronize()
        t = defaultdict(float)
        for name, s, e in self._pending:
            t[name] += s.elapsed_time(e)
        self._pending = []
        if keep:
            self.frames.append(dict(t))


def infer(args):
    import time

    cfg = C.load_cfg()
    model, _ = C.build_model(cfg)
    C.apply_precision(model, args.mode)
    model.cuda().eval()
    C.set_queue_variant("matmul")
    timer = EventTimer()
    install_head_wrappers(model, timer)
    _, loader = C.build_test_loader(cfg)
    samples = C.prefetch(loader, args.warmup + args.iters)
    lat = []
    with torch.no_grad():
        for i, d in enumerate(samples):
            d = C.to_gpu(d)
            torch.cuda.synchronize()
            t0 = time.perf_counter()
            model(return_loss=False, rescale=True, **d)
            timer.end_frame(keep=i >= args.warmup)
            if i >= args.warmup:
                lat.append((time.perf_counter() - t0) * 1000)
    names = sorted({n for f in timer.frames for n in f})
    seg = {n: float(np.mean([f.get(n, 0.0) for f in timer.frames])) for n in names}
    C.dump_json(dict(mode=args.mode, queue="matmul", n_frames=len(timer.frames),
                     latency_ms_mean=float(np.mean(lat)), segments_ms_mean=seg),
                os.path.join(C.OUT_ROOT, f"exp5prep_head_infer_time_{args.mode}.json"))


def head_dims(cfg):
    out = {}
    for prefix, attr in HEADS:
        h = cfg.model.head[attr]
        order = h.operation_order
        out[prefix] = dict(
            queries=h.instance_bank.num_anchor,
            temporal_queries=h.instance_bank.num_temp_instances,
            decoder_layers=order.count("refine"),
            ops_per_type={op: order.count(op) for op in sorted(set(order))},
            embed_dims=h.instance_bank.embed_dims,
            ffn=dict(in_channels=h.ffn.in_channels, hidden=h.ffn.feedforward_channels,
                     out=h.ffn.embed_dims),
            self_attn_embed=h.graph_model.embed_dims, temporal_attn_embed=h.temp_graph_model.embed_dims,
            attn_heads=h.graph_model.num_heads, decouple_attn=h.decouple_attn,
            deformable=dict(groups=h.deformable_model.num_groups, levels=h.deformable_model.num_levels,
                            cams=h.deformable_model.num_cams),
        )
        if prefix == "map":
            out[prefix]["points_per_query"] = h.instance_bank.anchor_handler.get("num_sample", None) \
                or h.anchor_encoder.num_sample
    return out


# Torch-Pruning structural judgement (from the code, NOT traced with the library:
# torch-pruning is not installed and no pruning is run at this step).
PRUNABILITY = {
    "ffn": ("가능", "Linear(in→hidden)+ReLU+Linear(hidden→256) 표준 MLP. 은닉 차원(1024)만 자르면 "
                    "잔차 스트림(256)에 영향 없음. 커스텀 CUDA 없음"),
    "refine": ("내부 차원만 가능", "linear_relu_ln(256) 블록들. 입력(잔차 256)·출력(box 11 / cls / quality)은 "
                               "고정, 블록 사이 내부 차원은 가능. 커스텀 CUDA 없음"),
    "anchor_enc": ("내부 차원만 가능", "pos/size/yaw/vel 각 linear_relu_ln. 마지막 출력은 cat되어 query_pos(256)가 "
                                   "되므로 고정, 내부 차원은 가능. 커스텀 CUDA 없음"),
    "self_attn": ("어려움", "FlashMHA가 nn.Linear가 아닌 원시 in_proj_weight(3E×E)를 chunk해서 쓰고, "
                         "flash-attn CUDA 커널(head_dim 제약) 호출. Torch-Pruning 기본 추적 불가 → 사용자 정의 pruner 필요. "
                         "fc_before/fc_after(decouple)는 일반 Linear"),
    "temporal_attn": ("어려움", "self_attn과 같은 모듈(FlashMHA + flash-attn 커널)"),
    "deformable": ("불가에 가까움", "커스텀 CUDA deformable_aggregation이 FPN 특징맵 채널(256)·그룹(8)과 묶여 있음. "
                                 "weights_fc/output_proj/camera_encoder는 Linear지만 채널을 자르면 커널 입력 형태가 바뀜"),
    "norm": ("잔차 스트림에 묶임", "LayerNorm(256 / FFN 입력 512). 잔차 스트림 폭을 바꾸지 않는 한 고정"),
    "residual_stream": ("불가", "embed 256은 det/map/motion head, DAF, FPN 출력, teacher KD 특징에 공통 → 자르면 전부 연쇄 변경"),
}


def summarize(args):
    cfg = C.load_cfg()
    train = json.load(open(os.path.join(C.OUT_ROOT, "expB_train_bs4_cpon_heads_matmul.json")))
    tmod = train["modules"]
    inf = json.load(open(os.path.join(C.OUT_ROOT, f"exp5prep_head_infer_time_{args.mode}.json")))
    tseg = inf["segments_ms_mean"]
    dims = head_dims(cfg)

    def cat_of(name):
        parts = name.split(".")
        return ".".join(parts[2:])

    result = dict(dims=dims, prunability=PRUNABILITY, heads={}, ffn_pruning={})
    for prefix, _ in HEADS:
        names = [n for n in tmod if n.startswith(prefix + ".") and n != f"{prefix}.total"]
        layers = sorted({n.split(".")[1] for n in names}, key=lambda x: (x != "pre", x))
        table = {}
        for L in layers:
            row = {}
            for cat in CATEGORIES[:-1] + ["ffn.hidden", "ffn.out_proj"]:
                n = f"{prefix}.{L}.{cat}"
                if n in tmod or n in tseg:
                    row[cat] = dict(train_retained_MB=tmod.get(n, {}).get("retained_MB_mean", 0.0),
                                    train_transient_MB=tmod.get(n, {}).get("transient_MB_max", 0.0),
                                    infer_ms=tseg.get(n, 0.0))
            table[L] = row
        tot_ret = tmod[f"{prefix}.total"]["retained_MB_mean"]
        tot_ms = tseg[f"{prefix}.total"]
        cats = {}
        for cat in CATEGORIES[:-1] + ["ffn.hidden", "ffn.out_proj"]:
            cats[cat] = dict(
                train_retained_MB=sum(r[cat]["train_retained_MB"] for r in table.values() if cat in r),
                infer_ms=sum(r[cat]["infer_ms"] for r in table.values() if cat in r))
        top = CATEGORIES[:-1]
        cats["other"] = dict(train_retained_MB=tot_ret - sum(cats[c]["train_retained_MB"] for c in top),
                             infer_ms=tot_ms - sum(cats[c]["infer_ms"] for c in top))
        for c in cats.values():
            c["train_share"] = c["train_retained_MB"] / tot_ret
            c["infer_share"] = c["infer_ms"] / tot_ms
        result["heads"][prefix] = dict(total=dict(train_retained_MB=tot_ret,
                                                  train_transient_MB=tmod[f"{prefix}.total"]["transient_MB_max"],
                                                  infer_ms=tot_ms),
                                       categories=cats, per_layer=table)

    # FFN hidden-width pruning estimate (ratio r of the hidden channels):
    #   memory: tensors that carry the hidden dim (ffn.hidden retained: ReLU output,
    #           dropout mask/output kept for Linear2) shrink by r
    #   time  : upper bound r * (ffn.hidden + ffn.out_proj) inference time, assuming
    #           GEMM time scales with the hidden width (small GEMMs often scale less)
    act_total = train["activations_MB"]["mean"]
    peak = train["peak_allocated_MB_max"]
    lat = inf["latency_ms_mean"]
    for r in (0.2, 0.4):
        est = {}
        mem = sum(result["heads"][p]["categories"]["ffn.hidden"]["train_retained_MB"] for p, _ in HEADS) * r
        tms = sum(result["heads"][p]["categories"]["ffn.hidden"]["infer_ms"]
                  + result["heads"][p]["categories"]["ffn.out_proj"]["infer_ms"] for p, _ in HEADS) * r
        for p, _ in HEADS:
            c = result["heads"][p]["categories"]
            est[p] = dict(train_mem_saved_MB=c["ffn.hidden"]["train_retained_MB"] * r,
                          head_mem_share=c["ffn.hidden"]["train_retained_MB"] * r / result["heads"][p]["total"]["train_retained_MB"],
                          infer_ms_saved_upper=(c["ffn.hidden"]["infer_ms"] + c["ffn.out_proj"]["infer_ms"]) * r,
                          head_time_share_upper=(c["ffn.hidden"]["infer_ms"] + c["ffn.out_proj"]["infer_ms"]) * r
                          / result["heads"][p]["total"]["infer_ms"])
        est["both"] = dict(train_mem_saved_MB=mem, of_total_activations=mem / act_total, of_peak_allocated=mem / peak,
                           infer_ms_saved_upper=tms, of_frame_latency_upper=tms / lat)
        result["ffn_pruning"][f"{int(r * 100)}%"] = est
    result["reference"] = dict(train_activations_MB_mean=act_total, train_peak_allocated_MB=peak,
                               infer_latency_ms=lat, infer_mode=args.mode)
    C.dump_json(result, os.path.join(C.OUT_ROOT, "exp5prep_head_breakdown.json"))
    for p, _ in HEADS:
        h = result["heads"][p]
        print(f"== {p}: train retained {h['total']['train_retained_MB']:.0f} MB, infer {h['total']['infer_ms']:.2f} ms")
        for c, v in h["categories"].items():
            print(f"   {c:<14} {v['train_retained_MB']:8.1f} MB ({v['train_share']*100:5.1f}%)  "
                  f"{v['infer_ms']:7.2f} ms ({v['infer_share']*100:5.1f}%)")
    for k, v in result["ffn_pruning"].items():
        print(k, {kk: {a: round(b, 4) for a, b in vv.items()} for kk, vv in v.items()})


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--infer", action="store_true")
    ap.add_argument("--summarize", action="store_true")
    ap.add_argument("--mode", choices=C.PRECISION_MODES, default="repo")
    ap.add_argument("--warmup", type=int, default=10)
    ap.add_argument("--iters", type=int, default=100)
    args = ap.parse_args()
    torch.multiprocessing.set_start_method("fork")
    if args.infer:
        infer(args)
    if args.summarize:
        summarize(args)
