import torch
from torch import nn
import torch.nn.functional as F


class KDLoss(nn.Module):
    def __init__(self, T=1):
        super(KDLoss, self).__init__()
        self.T = T

    def forward(self, our_emb, target_emb):
        soft_targets = nn.functional.softmax(target_emb / self.T, dim=-1)
        soft_prob = nn.functional.log_softmax(our_emb / self.T, dim=-1)

        soft_targets_loss = torch.sum(soft_targets * (soft_targets.log() - soft_prob)) / soft_prob.size()[0] * (self.T ** 2)
        return soft_targets_loss


class ContrastiveLoss(nn.Module):
    def __init__(self, batch_size, temperature=0.5):
        super().__init__()
        self.batch_size = batch_size
        self.register_buffer("temperature", torch.tensor(temperature))
        self.register_buffer("negatives_mask", (~torch.eye(batch_size * 2, batch_size * 2, dtype=bool)).float())

    def forward(self, emb_target, emb_surrogate):
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        representations = torch.cat([emb_target, emb_surrogate], dim=0)
        similarity_matrix = F.cosine_similarity(representations.unsqueeze(1), representations.unsqueeze(0), dim=2)
        sim_ij = torch.diag(similarity_matrix, self.batch_size)
        sim_ji = torch.diag(similarity_matrix, -self.batch_size)
        positives = torch.cat([sim_ij, sim_ji], dim=0)
        nominator = torch.exp(positives / self.temperature)
        denominator = self.negatives_mask.to(device) * torch.exp(similarity_matrix / self.temperature.to(device))
        loss_partial = -torch.log(nominator / torch.sum(denominator, dim=1))
        loss = (torch.sum(loss_partial)) / (2 * self.batch_size)
        return loss.mean()


class ContKDLoss(nn.Module):
    def __init__(self, batch_size, temperature=0.5, kd_T=1, kd_weight=1):
        super().__init__()
        self.batch_size = batch_size
        self.register_buffer("temperature", torch.tensor(temperature))
        self.register_buffer("negatives_mask", (~torch.eye(batch_size * 2, batch_size * 2, dtype=bool)).float())
        self.T = kd_T
        self.kd_weight = kd_weight

    def forward(self, emb_target, emb_surrogate):
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        # device = torch.device("cpu")
        # ----------  CONT-STEAL -------------
        representations = torch.cat([emb_target, emb_surrogate], dim=0)
        similarity_matrix = F.cosine_similarity(representations.unsqueeze(1), representations.unsqueeze(0), dim=2)
        sim_ij = torch.diag(similarity_matrix, self.batch_size)
        sim_ji = torch.diag(similarity_matrix, -self.batch_size)
        positives = torch.cat([sim_ij, sim_ji], dim=0)
        nominator = torch.exp(positives / self.temperature)
        denominator = self.negatives_mask.to(device) * torch.exp(similarity_matrix / self.temperature.to(device))
        loss_partial = -torch.log(nominator / torch.sum(denominator, dim=1))
        loss = (torch.sum(loss_partial)) / (2 * self.batch_size)
        cont_loss = loss.mean()

        # ----------- KD ---------------
        soft_targets = nn.functional.softmax(emb_target / self.T, dim=-1)
        soft_prob = nn.functional.log_softmax(emb_surrogate / self.T, dim=-1)
        kd_loss = torch.sum(soft_targets * (soft_targets.log() - soft_prob)) / soft_prob.size()[0] * (self.T ** 2)

        mse = nn.functional.mse_loss(emb_surrogate.float(), emb_target.float())

        return cont_loss + self.kd_weight * kd_loss + mse
