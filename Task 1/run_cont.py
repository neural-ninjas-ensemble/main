import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision.transforms import Compose, PILToTensor, RandomHorizontalFlip, ColorJitter

from loss_functions import ContrastiveLoss
from custom_model import Encoder
from taskdataset import TaskDataset
from dataset_merger import DatasetMerger
from train import train_epoch
from test import eval
from utils import save_model, save_history, get_position_by_id

import numpy as np
import pandas as pd


def main():
    torch.manual_seed(42)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    BATCH_SIZE = 128
    EPOCHS = 30
    LR = 0.001

    dataset1 = torch.load("./data/ModelStealing.pt")
    dataset1.transform = Compose([
        PILToTensor(),
        RandomHorizontalFlip(p=0.5),
        ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1, hue=0.1),
    ])

    ids = pd.read_csv("./data/ids2000.csv")["id"]
    ids = get_position_by_id(ids.values, dataset1)
    subset_dataset1 = torch.utils.data.Subset(dataset1, ids)

    dataset = DatasetMerger(subset_dataset1, "./data/TargetEmbeddings2000.pt")

    train_size = int(0.8 * len(dataset))
    test_size = len(dataset) - train_size
    train_dataset, test_dataset = torch.utils.data.random_split(dataset, [train_size, test_size])

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, drop_last=True)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=True, drop_last=True)

    model = Encoder().to(device)
    optimizer = optim.Adam(model.parameters(), lr=LR)
    # criterion = ContrastiveLoss(BATCH_SIZE)
    criterion = nn.MSELoss()

    # TRAINING
    history = np.zeros((EPOCHS, 2))
    for epoch in range(EPOCHS):
        train_epoch(device, model, criterion, optimizer, train_loader)
        loss, l2_loss = eval(device, epoch, model, criterion, test_loader)

        history[epoch, 0] = loss
        history[epoch, 1] = l2_loss

    # SAVE SCORE HISTORY
    save_history(history, "cont_loss")

    # SAVE MODEL
    save_model(model)


if __name__ == '__main__':
    main()
