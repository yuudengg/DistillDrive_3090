# Second training run of recover_stage2_distribution_det6_map4.py with a different seed (train with --seed 1).
# Same settings; only the work_dir (and therefore resume_from) differs so it never resumes the seed-0 run.
_base_ = ["./recover_stage2_distribution_det6_map4.py"]
import os

_latest = "work_dirs/recover_stage2_distribution_det6_map4_seed1/latest.pth"
resume_from = _latest if os.path.exists(_latest) else None
