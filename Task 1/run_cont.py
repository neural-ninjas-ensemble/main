import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision.transforms import Compose, PILToTensor

from loss_functions import ContrastiveLoss
from custom_model import Encoder
from taskdataset import TaskDataset
from dataset_merger import DatasetMerger
from train import train_epoch
from test import eval
from utils import save_model, save_history

import numpy as np
import pandas as pd


def main():
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    BATCH_SIZE = 8
    EPOCHS = 10
    LR = 3e-4

    ids = pd.read_csv("./data/ids.csv")["id"]
    dataset1 = torch.load("./data/ModelStealingPub.pt")
    dataset1.transform = Compose([
        PILToTensor(),
    ])
    subset_dataset1 = torch.utils.data.Subset(dataset1, ids)

    dataset = DatasetMerger(subset_dataset1, "./data/TargetEmbeddings.pt")

    train_size = int(0.8 * len(dataset))
    test_size = len(dataset) - train_size
    train_dataset, test_dataset = torch.utils.data.random_split(dataset, [train_size, test_size])

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=True)

    model = Encoder().to(device)
    optimizer = optim.Adam(model.parameters(), lr=LR)
    criterion = ContrastiveLoss(BATCH_SIZE)

    # TRAINING
    history = np.zeros((EPOCHS, 2))
    for epoch in range(EPOCHS):
        train_epoch(device, model, criterion, optimizer, train_loader)
        loss, l2_loss = eval(device, epoch, model, criterion, test_loader)

        history[epoch, 0] = loss
        history[epoch, 1] = l2_loss
        break

    # SAVE SCORE HISTORY
    save_history(history, "cont_loss")

    # SAVE MODEL
    save_model(model)


if __name__ == '__main__':
    main()
