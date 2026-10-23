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
    if deploy:
        assert all(k.startswith("head.distillation_head.") for k in unexpected), unexpected[:10]
    else:
        assert not unexpected, unexpected[:10]
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
