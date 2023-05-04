import torch.nn as nn


class SimpleNN(nn.Module):
    def __init__(self):
        super(SimpleNN, self).__init__()
        self.network = nn.Sequential(
            nn.Linear(40, 64),
            nn.LeakyReLU(),
            nn.Linear(64, 32),
            nn.LeakyReLU(),
            nn.Linear(32, 4),
            nn.LeakyReLU(),
            nn.Linear(4, 2)
        )

    def forward(self, x):
        predict = self.network(x)
        return predict
