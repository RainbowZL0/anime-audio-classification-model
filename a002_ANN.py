import torch.utils.data
from torch import nn, optim
from torch.utils.data import DataLoader

from a000_configuration import *
from a001_preprocess import *
from a002_网络基类 import *


class SimpleNN(MyNN):
    def __init__(self):
        super().__init__()

        self.network = nn.Sequential(
            nn.Linear(40, 64),
            nn.LeakyReLU(),
            nn.Linear(64, 32),
            nn.LeakyReLU(),
            nn.Linear(32, 16),
            nn.LeakyReLU(),
            nn.Linear(16, 8),
            nn.LeakyReLU(),
            nn.Linear(8, 4),
            nn.LeakyReLU(),
            nn.Linear(4, 2)
        )
        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = optim.Adam(self.parameters(), lr=LR, weight_decay=1e-6)

    def forward(self, x):
        predict = self.network(x)
        return predict


if __name__ == '__main__':




    pass
