import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision.transforms import Compose, PILToTensor, RandomHorizontalFlip, ColorJitter

from loss_functions import ContKDLoss
from custom_model import Encoder, Identity
from taskdataset import TaskDataset
from dataset_merger import DatasetMerger
from train import train_epoch, train_epoch_pretr
from test import eval, eval_pretr
from utils import save_model, save_history, get_position_by_id

import numpy as np
import pandas as pd
from datetime import datetime
from copy import deepcopy


def main():
    torch.manual_seed(42)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    # device = torch.device("cpu")

    BATCH_SIZE = 128
    PRETRAINING_EPOCHS = 2
    EPOCHS = 20
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
    full_loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True, drop_last=True)

    model = Encoder().to(device)
    pretr_criterion = nn.CrossEntropyLoss()
    pretr_optimizer = optim.Adam(model.parameters(), lr=LR)

    # PRETRAINING
    print("PRETRAINING")
    for epoch in range(PRETRAINING_EPOCHS):
        train_epoch_pretr(device, model, pretr_criterion, pretr_optimizer, full_loader)
        eval_pretr(device, epoch, model, pretr_criterion, test_loader)

    sd = deepcopy(model.state_dict())

    training(sd, device, BATCH_SIZE, LR, EPOCHS, train_loader, test_loader)


def training(sd, device, BATCH_SIZE, LR, EPOCHS, train_loader, test_loader):
    model = Encoder()
    model.load_state_dict(sd)
    model.fc = Identity()
    model = model.to(device)
    optimizer = optim.Adam(model.parameters(), lr=LR)
    criterion = ContKDLoss(BATCH_SIZE, temperature=0.5, kd_T=2, kd_weight=5)
    # TRAINING
    print("TRAINING")
    now = datetime.now()
    hour = f"{now.hour}:{now.minute}"

    best_loss = 1000.
    history = np.zeros((EPOCHS, 2))
    for epoch in range(EPOCHS):
        train_epoch(device, model, criterion, optimizer, train_loader)
        loss, l2_loss = eval(device, epoch, model, criterion, test_loader)

        if l2_loss < best_loss:
            best_loss = l2_loss
            save_model(model, hour)

        history[epoch, 0] = loss
        history[epoch, 1] = l2_loss

    # SAVE SCORE HISTORY
    save_history(history, "kd_cont_pretr_loss", hour)


if __name__ == '__main__':
    main()
