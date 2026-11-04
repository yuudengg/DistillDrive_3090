# C: recovery fine-tuning of the 4-layer det/map decoder (stage2_distribution_layers4),
# initialised from the 6-layer stage2 checkpoint (dropped layers' keys are ignored at load).
_base_ = ["./stage2_distribution_layers4.py"]
import os

# base config re-assigns load_from to stage1_soap.pth on its last line -> must override
load_from = "checkpoint/distilldrive_stage2_distribution.pth"
_latest = "work_dirs/recover_stage2_distribution_layers4/latest.pth"
resume_from = _latest if os.path.exists(_latest) else None
# base computes checkpoint_config from checkpoint_epoch_interval at its own parse time -> set directly
checkpoint_config = dict(interval=7032)
runner = dict(type="IterBasedRunner", max_iters=7032 * 3)
optimizer = dict(lr=1e-4)
# skip in-training eval (~2.4h each): interval > max_iters so EvalHook never fires; evaluate afterwards with queue_matmul_eval.py --ckpt
evaluation = dict(interval=7032 * 3 + 1)
