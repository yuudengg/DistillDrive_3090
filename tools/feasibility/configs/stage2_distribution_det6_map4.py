# det_head stopped after decoder layer 6, map_head after layer 4 (of 6). No retraining;
# see stage2_distribution_layers4.py for what reads the last executed layer.
_base_ = ["../../../projects/configs/stage2/distilldrive_stage2_distribution.py"]

det_head_layers = 6
map_head_layers = 4
_det_first = ["deformable", "ffn", "norm", "refine"]
_map_first = ["gnn", "norm", "deformable", "ffn", "norm", "refine"]
_later = ["temp_gnn", "gnn", "norm", "deformable", "ffn", "norm", "refine"]

model = dict(
    head=dict(
        det_head=dict(operation_order=_det_first + _later * (det_head_layers - 1)),
        map_head=dict(operation_order=_map_first + _later * (map_head_layers - 1)),
    )
)
