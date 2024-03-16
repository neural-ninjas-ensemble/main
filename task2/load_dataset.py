import torch
from taskdataset import TaskDataset


if __name__ == "__main__":
    dataset = torch.load("/home/hack33/task2/SybilAttack.pt")

    print(dataset.ids)#, dataset.imgs, dataset.labels)
    print(type(dataset))
    print(len(dataset))
