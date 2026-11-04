# D: control for C - same 3-epoch recovery fine-tuning, but with the original 6-layer decoder.
# Not inherited from recover_..._layers4.py so resume_from points at this experiment's own work_dir.
_base_ = ["../../../projects/configs/stage2/distilldrive_stage2_distribution.py"]
import os

# base config re-assigns load_from to stage1_soap.pth on its last line -> must override
load_from = "checkpoint/distilldrive_stage2_distribution.pth"
_latest = "work_dirs/recover_stage2_distribution_layers6/latest.pth"
resume_from = _latest if os.path.exists(_latest) else None
# base computes checkpoint_config from checkpoint_epoch_interval at its own parse time -> set directly
checkpoint_config = dict(interval=7032)
runner = dict(type="IterBasedRunner", max_iters=7032 * 3)
optimizer = dict(lr=1e-4)
# skip in-training eval (~48min): interval > max_iters so EvalHook never fires; evaluate afterwards with queue_matmul_eval.py --ckpt
evaluation = dict(interval=7032 * 3 + 1)
