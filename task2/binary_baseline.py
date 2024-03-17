import numpy as np
import torch
from taskdataset import TaskDataset

if __name__ == '__main__':
    dataset = torch.load('/home/hack33/task2/SybilAttack.pt')
    id_order = dataset.ids
    reps_in_order = [[0] * 384 for i in id_order]

    np.savez(
        "submission_binary_baseline.npz",
        ids=id_order,
        representations=reps_in_order,
    )