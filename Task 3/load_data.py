import numpy as np

from torch.utils.data import Dataset
from typing import Tuple
import torch
from torch.utils.data import DataLoader

class Task3Dataset(Dataset):
    def __init__(self, transform=None):

        self.train = np.load("data/DefenseTransformationEvaluate.npz")
        self.test = np.load("data/DefenseTransformationSubmit.npz")
        self.train_data = self.train["representations"]
        self.train_labels = self.train["labels"]

        # self.transform = transform

    def __getitem__(self, index) -> Tuple[torch.Tensor, int]:
        data = self.train_data[index]
        label = self.train_labels[index]
        # if not self.transform is None:
        #     img = self.transform(img)
        # label = self.labels[index]
        return data, label

    def __len__(self):
        return len(self.train)

if __name__ == "__main__":
    data = np.load(
        "data/DefenseTransformationEvaluate.npz"
    )

    print(set(data["labels"]))
    # print(data["labels"][0], data["representations"][0])

    # data = np.load("data/DefenseTransformationSubmit.npz")
    # print(data["representations"].shape)

    # dataset = Task3Dataset()
    # data_loaded = DataLoader(dataset, batch_size=32, shuffle=True)
    # print(data_loaded)