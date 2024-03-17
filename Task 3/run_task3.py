import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision.transforms import Compose, PILToTensor
from torch import nn
from train import train_epoch
# from test import eval
from utils import save_model, save_history, get_position_by_id
from load_data import *
import numpy as np
import pandas as pd
from model import *


def main():
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    BATCH_SIZE = 8
    EPOCHS = 10
    LR = 3e-4

    dataset = Task3Dataset()
    # dataset1.transform = Compose([
    #     PILToTensor(),
    # ])

    # ids = pd.read_csv("./data/ids500.csv")["id"]
    # ids = get_position_by_id(ids.values, dataset1)
    # subset_dataset1 = torch.utils.data.Subset(dataset1, ids)

    # dataset = DatasetMerger(subset_dataset1, "./data/TargetEmbeddings500.pt")

    train_size = int(0.8 * len(dataset))
    test_size = len(dataset) - train_size
    train_dataset, test_dataset = torch.utils.data.random_split(dataset, [train_size, test_size])

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, drop_last=True)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=True, drop_last=True)

    model = Net().to(device)
    optimizer = optim.Adam(model.parameters(), lr=LR)
    criterion = nn.MSELoss(BATCH_SIZE)

    # TRAINING
    history = np.zeros((EPOCHS, 2))
    for epoch in range(EPOCHS):
        train_epoch(device, model, criterion, optimizer, train_loader)
        loss, l2_loss = eval(device, epoch, model, criterion, test_loader)

        history[epoch, 0] = loss
        history[epoch, 1] = l2_loss
        break

    # SAVE SCORE HISTORY
    save_history(history, "mse_loss")

    # SAVE MODEL
    save_model(model)

if __name__ == '__main__':
    main()
