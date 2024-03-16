import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision.transforms import Compose, PILToTensor

from contrastive_loss import ContrastiveLoss
from knowledge_distillation import KDLoss
from custom_model import Encoder
from taskdataset import TaskDataset
from dataset_merger import DatasetMerger
from train import train_epoch
from test import eval
from utils import save_model, save_history

import numpy as np


def main():
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    # RESNET CODE --------
    # model = resnet18(weights=ResNet18_Weights.IMAGENET1K_V1)
    # encoder = torch.nn.Sequential(*(list(model.children())[:-1]))
    # print(encoder)
    # print(sum(p.numel() for p in encoder.parameters() if p.requires_grad))
    # --------------------

    BATCH_SIZE = 1
    EPOCHS = 10
    LR = 3e-4

    dataset1 = torch.load("./data/ModelStealing.pt")
    dataset1.transform = Compose([
        PILToTensor(),
    ])
    dataset = DatasetMerger(dataset1, "./data/TargetEmbeddings.pt")

    train_loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)

    model = Encoder().to(device)
    optimizer = optim.Adam(model.parameters(), lr=LR)
    criterion = KDLoss(T=2)

    # TRAINING
    history = np.zeros((EPOCHS, 2))
    for epoch in range(EPOCHS):
        train_epoch(device, model, criterion, optimizer, train_loader)
        cont_loss, l2_loss = eval(device, epoch, model, criterion, train_loader)

        history[epoch, 0] = cont_loss
        history[epoch, 1] = l2_loss

    # SAVE SCORE HISTORY
    save_history(history)

    # SAVE MODEL
    save_model(model)


if __name__ == '__main__':
    main()
