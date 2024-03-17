from typing import Tuple
import torch
from torch.utils.data import Dataset
from utils import get_class_dict


class DatasetMerger(Dataset):
    def __init__(self, dataset1, path_to_target_tensors):
        self.dataset1 = dataset1
        self.target_tensors = torch.load(path_to_target_tensors)
        self.class_dict = get_class_dict()

    def __getitem__(self, index) -> Tuple[int, torch.Tensor, int, torch.Tensor]:
        id_, img, label = self.dataset1[index]

        target_embedding = self.target_tensors[index]
        if img.shape != torch.Size([3, 32, 32]):
            img = torch.cat((img, img, img), dim=0)
        label = self.class_dict[label]
        return id_, img, label, target_embedding

    def __len__(self):
        return len(self.dataset1)