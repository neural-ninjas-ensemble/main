import torch
import torch.nn as nn


class Encoder(nn.Module):
    def __init__(self):
        super(Encoder, self).__init__()

        self.block = nn.Sequential(
            nn.Conv2d(3, 8, kernel_size=7, stride=1, padding='same', bias=False),
            nn.BatchNorm2d(8),
            nn.GELU(),
            nn.Dropout2d(0.4),

            nn.Conv2d(8, 16, kernel_size=3, stride=1, padding='same', bias=False),
            nn.BatchNorm2d(16),
            nn.GELU(),
            nn.MaxPool2d(2),
            nn.Dropout2d(0.25),

            nn.Conv2d(16, 16, kernel_size=3, stride=1, padding='same', bias=False),
            nn.BatchNorm2d(16),
            nn.GELU(),
            nn.Dropout2d(0.25),

            nn.Conv2d(16, 32, kernel_size=3, stride=1, padding='same', bias=True),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),

            nn.Flatten(),
            nn.Linear(2048, 512)
        )

    def forward(self, x):
        x = self.block(x)
        return x


class Identity(nn.Module):
    def __init__(self):
        super(Identity, self).__init__()

    def forward(self, x):
        return x
