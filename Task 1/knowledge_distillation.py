import torch
import torch.nn as nn


class KDLoss(nn.Module):
    def __init__(self, T=1):
        super(KDLoss, self).__init__()
        self.T = T

    def forward(self, our_emb, target_emb):
        soft_targets = nn.functional.softmax(target_emb / self.T, dim=-1)
        soft_prob = nn.functional.log_softmax(our_emb / self.T, dim=-1)

        soft_targets_loss = torch.sum(soft_targets * (soft_targets.log() - soft_prob)) / soft_prob.size()[0] * (self.T ** 2)
        return soft_targets_loss
