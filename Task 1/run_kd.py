import torch
import pandas as pd
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision.transforms import Compose, PILToTensor

from loss_functions import KDLoss
from custom_model import Encoder
from taskdataset import TaskDataset
from dataset_merger import DatasetMerger
from train import train_epoch
from test import eval
from utils import save_model, save_history, get_position_by_id

import numpy as np


def main():
    torch.manual_seed(42)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    BATCH_SIZE = 64
    EPOCHS = 30
    LR = 3e-4

    dataset1 = torch.load("./data/ModelStealing.pt")
    dataset1.transform = Compose([
        PILToTensor(),
    ])

    ids = pd.read_csv("./data/ids2000.csv")["id"]
    ids = get_position_by_id(ids.values, dataset1)
    subset_dataset1 = torch.utils.data.Subset(dataset1, ids)

    dataset = DatasetMerger(subset_dataset1, "./data/TargetEmbeddings2000.pt")

    train_size = int(0.8 * len(dataset))
    test_size = len(dataset) - train_size
    train_dataset, test_dataset = torch.utils.data.random_split(dataset, [train_size, test_size])

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=True)

    model = Encoder().to(device)
    optimizer = optim.Adam(model.parameters(), lr=LR)
    criterion = KDLoss(T=2)

    # TRAINING
    history = np.zeros((EPOCHS, 2))
    for epoch in range(EPOCHS):
        train_epoch(device, model, criterion, optimizer, train_loader)
        loss, l2_loss = eval(device, epoch, model, criterion, test_loader)

        history[epoch, 0] = loss
        history[epoch, 1] = l2_loss

    # SAVE SCORE HISTORY
    save_history(history, "kd_loss")

    # SAVE MODEL
    save_model(model)


if __name__ == '__main__':
    main()
