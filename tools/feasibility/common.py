"""Shared helpers for the pruning/quantization feasibility experiments.

Nothing here modifies configs, checkpoints or training code: every change
(deploy model without the teacher head, precision mode) is applied to the
in-memory config/model only.
"""
import copy
import functools
import importlib
import json
import os
import re
import sys

import torch

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from mmcv import Config  # noqa: E402
from mmcv.runner import wrap_fp16_model  # noqa: E402

DEFAULT_CONFIG = "projects/configs/stage2/distilldrive_stage2_distribution.py"
DEFAULT_CKPT = "checkpoint/distilldrive_stage2_distribution.pth"
OUT_ROOT = os.path.join(REPO_ROOT, "tools", "feasibility", "outputs")

# fp32     : no wrap_fp16_model -> everything fp32 (flash-attn is fp16 internally regardless)
# repo     : what tools/test.py does (cfg.fp16 set -> wrap_fp16_model):
#            extract_feat (backbone/neck) runs under autocast, heads fp32
# full_amp_raw : repo + head.forward under torch.autocast(fp16) as well.
#            DeformableAggregation casts its inputs to float itself, so that custom
#            CUDA op always runs fp32. Produces NaN from val frame 436 on (see REPORT.md).
# full_amp : full_amp_raw + fp32 for the two pieces that overflow fp16 at scene
#            changes (dt ~ -4.3e5 s): anchor_projection (anchor += vel * dt) and
#            anchor_encoder (first Linear sees those ~1e7 m coordinates; its
#            LayerNorm brings the embedding back into fp16 range)
PRECISION_MODES = ("fp32", "repo", "full_amp_raw", "full_amp")

MB = 1024 ** 2


def load_cfg(config=DEFAULT_CONFIG):
    os.chdir(REPO_ROOT)
    cfg = Config.fromfile(config)
    importlib.import_module("projects.mmdet3d_plugin")
    if cfg.get("cudnn_benchmark", False):
        torch.backends.cudnn.benchmark = True
    cfg.model.pretrained = None
    return cfg


def build_model(cfg, ckpt=DEFAULT_CKPT, deploy=True):
    """deploy=True drops distillation_head (teacher is only used by the KD loss)."""
    from mmdet.models import build_detector

    model_cfg = copy.deepcopy(cfg.model)
    if deploy:
        model_cfg.head.task_config.with_distillation = False
        model_cfg.train_cfg = None
    model = build_detector(model_cfg, test_cfg=cfg.get("test_cfg"))
    sd = torch.load(ckpt, map_location="cpu")["state_dict"]
    res = model.load_state_dict(sd, strict=False)
    missing, unexpected = res.missing_keys, res.unexpected_keys
    assert not missing, f"missing keys: {missing[:10]}"
    # a config with fewer decoder layers (tools/feasibility/configs) drops the trailing
    # head.<det|map>_head.layers.<i> entries; the remaining layers keep their weights
    n_layers = {h: len(getattr(model.head, h).layers) for h in ("det_head", "map_head")}

    def dropped_layer(k):
        m = re.match(r"head\.(det_head|map_head)\.layers\.(\d+)\.", k)
        return m is not None and int(m.group(2)) >= n_layers[m.group(1)]

    teacher = [k for k in unexpected if k.startswith("head.distillation_head.")]
    dropped = [k for k in unexpected if dropped_layer(k)]
    assert len(teacher) + len(dropped) == len(unexpected), \
        [k for k in unexpected if k not in teacher and k not in dropped][:10]
    assert deploy or not teacher, teacher[:10]
    return model, len(unexpected)


def _to_fp32(x):
    if torch.is_tensor(x):
        return x.float() if x.dtype == torch.float16 else x
    if isinstance(x, dict):
        return type(x)((k, _to_fp32(v)) for k, v in x.items())
    if isinstance(x, (list, tuple)):
        return type(x)(_to_fp32(v) for v in x)
    return x


def _fp32_call(fn):
    @functools.wraps(fn)
    def wrapped(*args, **kwargs):
        with torch.autocast(device_type="cuda", enabled=False):
            return fn(*_to_fp32(args), **_to_fp32(kwargs))

    return wrapped


def apply_precision(model, mode):
    assert mode in PRECISION_MODES, mode
    if mode != "fp32":
        wrap_fp16_model(model)
    if mode.startswith("full_amp"):
        # DistillDrive.forward is @force_fp32, which disables any outer autocast,
        # so autocast has to be re-entered at the head level.
        head_forward = model.head.forward

        @functools.wraps(head_forward)
        def amp_head_forward(*args, **kwargs):
            with torch.autocast(device_type="cuda", dtype=torch.float16):
                outs = head_forward(*args, **kwargs)
            # post_process is decorated force_fp32(apply_to=("model_outs")), whose
            # arg name never matches, so half outputs would leak into the decoders.
            return _to_fp32(outs)

        model.head.forward = amp_head_forward
    if mode == "full_amp":
        for head in (model.head.det_head, model.head.map_head):
            handler = head.instance_bank.anchor_handler
            handler.anchor_projection = _fp32_call(handler.anchor_projection)
            head.anchor_encoder.forward = _fp32_call(head.anchor_encoder.forward)
    return model


def _tracking_match(queue, det_output):
    """The tracking match matrix exactly as InstanceQueue.prepare_motion builds it."""
    match = det_output["instance_id"][..., None] == queue.prev_instance_id[:, None]
    if queue.tracking_threshold > 0:
        match = match * (queue.prev_confidence > queue.tracking_threshold).unsqueeze(1)
    return match


def prepare_motion_matmul(self, det_output, mask):
    """InstanceQueue.prepare_motion with the (B, N, N, C) broadcast-sum replaced by
    the equivalent matmul; everything else is line-for-line the original."""
    instance_feature = det_output["instance_feature"]
    det_anchors = det_output["prediction"][-1]
    if self.period == None:  # noqa: E711  (mirrors the original)
        self.period = instance_feature.new_zeros(instance_feature.shape[:2]).long()
    else:
        match = _tracking_match(self, det_output)
        # autocast off: a one-hot gather must not round fp32 queue entries to fp16
        with torch.autocast(device_type="cuda", enabled=False):
            for i in range(len(self.instance_feature_queue)):
                f = self.instance_feature_queue[i]
                self.instance_feature_queue[i] = torch.matmul(match.to(f.dtype), f)
                a = self.anchor_queue[i]
                self.anchor_queue[i] = torch.matmul(match.to(a.dtype), a)
        self.period = (match * self.period[:, None]).sum(dim=2)
    self.instance_feature_queue.append(instance_feature.detach())
    self.anchor_queue.append(det_anchors.detach())
    self.period += 1
    if len(self.instance_feature_queue) > self.queue_length:
        self.instance_feature_queue.pop(0)
        self.anchor_queue.pop(0)
    self.period = torch.clip(self.period, 0, self.queue_length)


def set_queue_variant(variant, match_stats=None):
    """variant: "original" | "matmul". With match_stats (dict), every call also
    records the match matrix dtype, its distinct values and max matches per row."""
    from projects.mmdet3d_plugin.models.motion.instance_queue import InstanceQueue

    if not hasattr(InstanceQueue, "_orig_prepare_motion"):
        InstanceQueue._orig_prepare_motion = InstanceQueue.prepare_motion
    impl = InstanceQueue._orig_prepare_motion if variant == "original" else prepare_motion_matmul
    if match_stats is None:
        InstanceQueue.prepare_motion = impl
        return

    match_stats.update(calls=0, dtypes=set(), values=set(), max_per_row=0, max_per_col=0)

    def recorded(self, det_output, mask):
        if self.period is not None:
            m = _tracking_match(self, det_output)
            match_stats["calls"] += 1
            match_stats["dtypes"].add(str(m.dtype))
            match_stats["values"].update(torch.unique(m.int()).tolist())
            match_stats["max_per_row"] = max(match_stats["max_per_row"], int(m.sum(-1).max()))
            match_stats["max_per_col"] = max(match_stats["max_per_col"], int(m.sum(-2).max()))
        return impl(self, det_output, mask)

    InstanceQueue.prepare_motion = recorded


def read_eval_layer(model, n_layers):
    """det_head / map_head post_process read decoder layer `n_layers` (1-based) instead of
    the last one. The lists are sliced instead of passing output_idx because the map
    decoder ignores output_idx (always uses [-1]). Only these two post_process calls are
    affected: the instance bank cache, instance ids, motion_plan_head and its
    post-processing still see the full model's last layer."""
    for attr in ("det_head", "map_head"):
        head = getattr(model.head, attr)
        orig = head.post_process

        def post_process(model_outs, output_idx=-1, _orig=orig):
            outs = dict(model_outs)
            for k in ("classification", "prediction", "quality"):
                if isinstance(outs.get(k), list):
                    outs[k] = outs[k][:n_layers]
            return _orig(outs, output_idx=-1)

        head.post_process = post_process
    return model


def tensor_bytes(tensors):
    return sum(t.numel() * t.element_size() for t in tensors)


def param_stats(module):
    params = list(module.parameters())
    return dict(
        params_M=sum(p.numel() for p in params) / 1e6,
        param_MB=tensor_bytes(params) / MB,
        buffer_MB=tensor_bytes(list(module.buffers())) / MB,
    )


def build_test_loader(cfg, workers=None, work_dir=None):
    from mmdet.datasets import build_dataset
    from mmdet.datasets import build_dataloader as build_dataloader_origin

    test_cfg = cfg.data.test
    test_cfg.test_mode = True
    test_cfg.pop("samples_per_gpu", None)
    if work_dir is not None:
        test_cfg.work_dir = work_dir
    dataset = build_dataset(test_cfg)
    loader = build_dataloader_origin(
        dataset,
        samples_per_gpu=1,
        workers_per_gpu=cfg.data.workers_per_gpu if workers is None else workers,
        dist=False,
        shuffle=False,
    )
    return dataset, loader


def to_gpu(data):
    from mmcv.parallel import scatter

    return scatter(data, [torch.cuda.current_device()])[0]


def prefetch(loader, n):
    """First n consecutive val frames, kept on CPU (temporal instance bank needs order)."""
    samples = []
    for i, data in enumerate(loader):
        samples.append(data)
        if len(samples) == n:
            break
    return samples


def dump_json(obj, path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as f:
        json.dump(obj, f, indent=2)
    print(f"[saved] {path}")
