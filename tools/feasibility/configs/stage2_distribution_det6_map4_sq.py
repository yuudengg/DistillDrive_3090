# E structure (det decoder 6 layers / map decoder 4 layers) + Sparse Query pruning of the planning
# decoder, for training-free evaluation of the E recovery checkpoints. The Sparse Query settings are
# copied unchanged from the teammate's projects/configs/stage2/distilldrive_stage2_distribution_sq.py
# (PR #2, merged into lightweighting as 83e39b6); only the planning head type and its pruning
# options differ from stage2_distribution_det6_map4.py. No training is run with this config.
_base_ = ["./stage2_distribution_det6_map4.py"]

num_query = 64
model = dict(
    head=dict(
        motion_plan_head=dict(
            type="SparseQueryMotionPlanningHead",
            num_query=num_query,
            min_keep_score=0.5,            # = HierarchicalPlanningDecoder.rescore score_thresh
            generative_hidden="preserve",  # released / recovered weights, no training
            preserve_rescore=True,
            force_positive=True,           # train only
            force_positive_radius=2.0,
            force_positive_prob=1.0,
            pruned_cls_fill=-60.0,
        )
    )
)
