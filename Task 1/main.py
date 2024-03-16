from torchvision.models import resnet18, ResNet18_Weights

import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision.transforms import Compose, PILToTensor
from torchvision.transforms.functional import pil_to_tensor

from contrastive_loss import ContrastiveLoss
from custom_model import Encoder
from taskdataset import TaskDataset
from dataset_merger import DatasetMerger
from train import train_epoch
from test import eval


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

    model = Encoder()
    optimizer = optim.Adam(model.parameters(), lr=LR)
    criterion = ContrastiveLoss(BATCH_SIZE)

    # TRAINING
    for epoch in range(EPOCHS):
        train_epoch(device, model, criterion, optimizer, train_loader)
        # cont_loss, l2_loss = eval(device, epoch, model, criterion, val_loader)


if __name__ == '__main__':
    main()
