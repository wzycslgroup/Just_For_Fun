import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable


class CVAE(nn.Module):
    def __init__(self, latent_dim, W, H):
        super().__init__()

        original_dim = W * H
        intermediate_dim = 256
        self.mean_class = nn.Linear(1, latent_dim)

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

    def compute_loss(self, y_pre, y_true):
        return F.binary_cross_entropy(y_pre, y_true, size_average=False) + self.kl_loss

    def generator(self, x):
        return self.decoder(x)

    def sampling(self, mean, log_var):
        eps = Variable(torch.randn(mean.shape)).cuda()
        return mean + eps * torch.exp(log_var / 2)

    def forward(self, x, label=None):
        h = self.dense_original(x)

        if label is None:
            mean_label = 0
        else:
            mean_label = self.mean_class(label.view(label.shape[0], 1))

        # compute mean and log_variance for z
        mean = self.linear_mean(h)
        log_var = self.linear_log_var(h)

        # sample z
        z = self.sampling(mean, log_var)

        # kl_loss (mean - mean_label)
        self.kl_loss = -0.5 * torch.sum(1 + log_var - (mean - mean_label).pow(2) - log_var.exp())

        return mean, self.decoder(z)