import torch
import torch.nn as nn
import torch.optim as optim

class SquareNet(nn.Module):
    def __init__(self):
        super(SquareNet, self).__init__()
        self.norm = nn.BatchNorm1d(1, affine=False)  # встроенная нормализация входа
        self.net = nn.Sequential(
            nn.Linear(1, 64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.Linear(64, 1)
        )

    def forward(self, x):
        x = self.norm(x)
        return self.net(x)