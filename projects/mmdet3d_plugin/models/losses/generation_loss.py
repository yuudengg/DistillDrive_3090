import torch
import torch.nn as nn
from mmdet.models import LOSSES

@LOSSES.register_module()
class ProbabilisticLoss(nn.Module):
    """
        kl-loss for present distribution and future distribution.
    """
    def __init__(self, loss_weight=1.0, only_valid_agent=False, ego_mode_num=18):
        super().__init__()
        self.loss_weight = loss_weight
        self.only_valid_agent = only_valid_agent
        self.ego_mode_num = ego_mode_num

    def forward(self, output, instance_gt_idx, gt_idx_bs):
        """
            output: dict of distribution
            instance_gt_idx: dict of matched instance
            gt_idx_bs: list of batch gt
        """
        present_mu = output['present_mu'] # [B, N1, 32]
        present_log_sigma = output['present_log_sigma'] # [B, N1, 32]
        future_mu = output['future_mu'] # [B, N1, 32]
        future_log_sigma = output['future_log_sigma'] # [B, N1, 32]

        var_future = torch.exp(2 * future_log_sigma)
        var_present = torch.exp(2 * present_log_sigma)
        
        BS, N1, C = present_mu.shape
        
        kl_loss_mask = torch.ones((BS, N1), dtype=torch.bool, device=present_mu.device)
        if self.only_valid_agent:
            indices = instance_gt_idx['indices']
            kl_loss_mask = torch.zeros((BS, N1), dtype=torch.bool, device=present_mu.device)
            present_mu_tmp = present_mu.new_zeros((BS, N1, C)).to(present_mu.device)
            present_log_sigma_tmp = present_log_sigma.new_zeros((BS, N1, C)).to(present_mu.device)
            var_present_tmp = var_present.new_zeros((BS, N1, C)).to(present_mu.device)
            future_mu_tmp = future_mu.new_zeros((BS, N1, C)).to(present_mu.device)
            future_log_sigma_tmp = future_log_sigma.new_zeros((BS, N1, C)).to(present_mu.device)
            var_future_tmp = var_future.new_zeros((BS, N1, C)).to(present_mu.device)
            for bs, (pred_idx, target_idx) in enumerate(indices):
                if target_idx is None or len(target_idx) == 0:
                    # Only ego Target
                    instance_idx = [n for n in range(N1 - self.ego_mode_num, N1)]
                    agent_idx = [n for n in range(N1 - self.ego_mode_num, N1)]
                else:
                    # future encoding gt agent id
                    gt_agent_idx = gt_idx_bs[bs]
                    # find mapping to instance
                    mapping_gt = [torch.where(target_idx == agent)[0].item() for agent in gt_agent_idx]
                    # ego feature concate in last
                    instance_idx = pred_idx[mapping_gt].tolist() + [n for n in range(N1 - self.ego_mode_num, N1)]
                    agent_idx = [i for i in range(len(gt_agent_idx))] + [n for n in range(N1 - self.ego_mode_num, N1)]

                kl_loss_mask[bs, agent_idx] = True
                # use idx to rewrite feature
                present_mu_tmp[bs, agent_idx] = present_mu[bs, instance_idx]
                present_log_sigma[bs, agent_idx] = present_log_sigma[bs, instance_idx]
                var_present_tmp[bs, agent_idx] = var_present[bs, instance_idx]
                future_mu_tmp[bs, agent_idx] = future_mu[bs, agent_idx]
                future_log_sigma_tmp[bs, agent_idx] = future_log_sigma[bs, agent_idx]
                var_future_tmp[bs, agent_idx] = var_future[bs, agent_idx]

            present_mu = present_mu_tmp
            present_log_sigma = present_log_sigma
            var_present = var_present_tmp
            future_mu = future_mu_tmp
            future_log_sigma = future_log_sigma_tmp
            var_future = var_future_tmp

        # VAE generative loss
        # var_present is 0 at positions left unfilled by the only_valid_agent
        # masking above; dividing by it there produces a 0/0 NaN in the
        # backward pass (even though kl_loss_mask excludes those positions
        # from the forward value), which corrupts gradients for every
        # parameter upstream of this loss. Clamp away from exact zero.
        var_present_safe = var_present.clamp(min=1e-6)
        kl_div = (
                present_log_sigma - future_log_sigma - 0.5 + (var_future + (future_mu - present_mu) ** 2) / (
                    2 * var_present_safe)
        ) # [B, N1, 32]

        kl_loss = torch.mean(torch.sum(kl_div[kl_loss_mask], dim=-1)) * self.loss_weight 
        # No available agent
        if torch.sum(kl_loss_mask) == 0:
            kl_loss.zero_()

        return {'distribution_loss': kl_loss}