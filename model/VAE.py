import torch
import torch.nn as nn
from torch.autograd import Variable


class VariationalAutoEncoder(nn.Module):
    def __init__(self, latent_dim, W, H):
        super().__init__()

        original_dim = W * H
        intermediate_dim = 256

        self.dense_original = nn.Sequential(
            nn.Linear(original_dim, intermediate_dim),
            nn.ReLU(),
            nn.Linear(intermediate_dim, intermediate_dim),
            nn.ReLU(),
        )

        self.linear_mean = nn.Linear(intermediate_dim, latent_dim)
        self.linear_log_var = nn.Linear(intermediate_dim, latent_dim)

        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, intermediate_dim),
            nn.ReLU(),
            nn.Linear(intermediate_dim, original_dim),
            nn.Sigmoid()
        )

        self.kl_loss = None
        self.loss_function = nn.MSELoss()

    def sampling(self, mu, log_var):
        eps = Variable(torch.randn(mu.shape)).cuda()
        return mu + eps * torch.exp(log_var / 2)

    def compute_loss(self, y_pre, y_true):
        return self.loss_function(y_pre, y_true) + self.kl_loss

    def generator(self, x):
        return self.decoder(x)

    def forward(self, x, label=None):
        h = self.dense_original(x)

        # compute mean and log_variance for z
        mean = self.linear_mean(h)
        log_var = self.linear_log_var(h)

        # sample z
        z = self.sampling(mean, log_var)

        # kl_loss
        self.kl_loss = -0.5 * torch.sum(1 + log_var - mean.pow(2) - log_var.exp())

        return mean, self.decoder(z)