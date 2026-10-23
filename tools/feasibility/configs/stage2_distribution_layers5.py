# det_head / map_head stopped after decoder layer 5 (of 6). No retraining: the first
# 5 layers keep their stage2 weights, the dropped layers' keys are ignored at load.
# instance_bank.update (after layer 1) is unchanged; instance_bank.cache, instance ids,
# motion_plan_head and post-processing read the last executed layer (classification[-1],
# prediction[-1], instance_feature, anchor_embed).
_base_ = ["../../../projects/configs/stage2/distilldrive_stage2_distribution.py"]

num_head_layers = 5
_det_first = ["deformable", "ffn", "norm", "refine"]
_map_first = ["gnn", "norm", "deformable", "ffn", "norm", "refine"]
_later = ["temp_gnn", "gnn", "norm", "deformable", "ffn", "norm", "refine"]

model = dict(
    head=dict(
        det_head=dict(operation_order=_det_first + _later * (num_head_layers - 1)),
        map_head=dict(operation_order=_map_first + _later * (num_head_layers - 1)),
    )
)
