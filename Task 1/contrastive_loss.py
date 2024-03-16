import torch
from torch import nn
import torch.nn.functional as F


class ContrastiveLoss(nn.Module):
    def __init__(self, batch_size, temperature=0.5):
        super().__init__()
        self.batch_size = batch_size
        self.temperature = temperature
        self.n_views = 1
        self.ce = nn.CrossEntropyLoss()

    def info_nce_loss(self, features, target_emb):
        labels = torch.cat([torch.arange(self.batch_size) for i in range(self.n_views)], dim=0)
        labels = (labels.unsqueeze(0) == labels.unsqueeze(1)).float()

        features = F.normalize(features, dim=1)

        similarity_matrix = torch.matmul(features, target_emb.T)

        # discard the main diagonal from both: labels and similarities matrix
        mask = torch.eye(labels.shape[0], dtype=torch.bool)
        labels = labels[~mask].view(labels.shape[0], -1)
        similarity_matrix = similarity_matrix[~mask].view(similarity_matrix.shape[0], -1)
        # assert similarity_matrix.shape == labels.shape

        # select and combine multiple positives
        positives = similarity_matrix[labels.bool()].view(labels.shape[0], -1)

        # select only the negatives the negatives
        negatives = similarity_matrix[~labels.bool()].view(similarity_matrix.shape[0], -1)

        logits = torch.cat([positives, negatives], dim=1)
        labels = torch.zeros(logits.shape[0], dtype=torch.long)

        logits = logits / self.temperature
        return logits, labels

    def forward(self, emb_target, emb_surrogate):
        logits, labels = self.info_nce_loss(emb_surrogate, emb_target)
        return self.ce(logits, labels)




# class ContrastiveLoss(nn.Module):
#     def __init__(self, batch_size, temperature=0.5):
#         super().__init__()
#         self.batch_size = batch_size
#         self.register_buffer("temperature", torch.tensor(temperature))
#         self.register_buffer("negatives_mask", (~torch.eye(batch_size * 2, batch_size * 2, dtype=bool)).float())
#
#     def forward(self, emb_target, emb_surrogate):
#         device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#         representations = torch.cat([emb_target, emb_surrogate], dim=0)
#         similarity_matrix = F.cosine_similarity(representations.unsqueeze(1), representations.unsqueeze(0), dim=2)
#         sim_ij = torch.diag(similarity_matrix, self.batch_size)
#         sim_ji = torch.diag(similarity_matrix, -self.batch_size)
#         positives = torch.cat([sim_ij, sim_ji], dim=0)
#         nominator = torch.exp(positives / self.temperature)
#         denominator = self.negatives_mask.to(device) * torch.exp(similarity_matrix / self.temperature.to(device))
#         loss_partial = -torch.log(nominator / torch.sum(denominator, dim=1))
#         loss = (torch.sum(loss_partial)) / (2 * self.batch_size)
#         return loss.mean()
