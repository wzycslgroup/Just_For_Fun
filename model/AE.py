import torch.nn as nn


class AutoEncoder(nn.Module):
    def __init__(self, dim, W, H):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(W*H, 2048),
            nn.Tanh(),
            nn.Linear(2048, 512),
            nn.Tanh(),
            nn.Linear(512, 128),
            nn.Tanh(),
            nn.Linear(128, 32),
            nn.Tanh(),
            nn.Linear(32, dim),
            # nn.Tanh(),
            # nn.Linear(16, dim),
        )
        self.decoder = nn.Sequential(
            nn.Linear(dim, 32),
            nn.Tanh(),
            nn.Linear(32, 128),
            nn.Tanh(),
            nn.Linear(128, 512),
            nn.Tanh(),
            nn.Linear(512, 2048),
            nn.Tanh(),
            nn.Linear(2048, W*H),
            nn.Sigmoid(),
        )
        self.loss_function = nn.MSELoss()

    def compute_loss(self, y_pre, y_true):
        return self.loss_function(y_pre, y_true)
        # return F.binary_cross_entropy(y_pre, y_true)

    def generator(self, x):
        return self.decoder(x)

    def forward(self, x, label=None):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return encoded, decoded
