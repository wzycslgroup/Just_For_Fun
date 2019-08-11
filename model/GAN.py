import torch
import torch.nn as nn
import torch.nn.functional as F


class Generator(nn.Module):
    def __init__(self, dim, n_class, W, H):
        super(Generator, self).__init__()
        self.dim = dim
        self.n_class = n_class
        self.W = W
        self.H = H
        self.G = nn.Sequential(
            nn.Linear(dim+n_class, 64),
            nn.LeakyReLU(),
            nn.Linear(64, 128),
            nn.LeakyReLU(),
            nn.Linear(128, 256),
            nn.LeakyReLU(),
            nn.Linear(256, 512),
            nn.LeakyReLU(),
            nn.Linear(512, W*H),
            nn.Tanh()
        )

    def forward(self, z, label):
        one_hot_y = torch.eye(self.n_class)[label.long()].cuda()
        z_y = torch.cat([z, one_hot_y], dim=1)
        return self.G(z_y)

    def compute_loss(self):
        loss = None
        return loss


class Discriminator(nn.Module):
    def __init__(self, W, H, n_class):
        super(Discriminator, self).__init__()
        self.W = W
        self.H = H
        self.n_class = n_class
        self.D = nn.Sequential(
            nn.Linear(W*H+n_class, 512),
            nn.LeakyReLU(),
            nn.Linear(512, 256),
            nn.LeakyReLU(),
            nn.Linear(256, 16),
            nn.LeakyReLU(),
            nn.Linear(16, 1),
            nn.Sigmoid()
        )

    def forward(self, img, label):
        one_hot_y = torch.eye(self.n_class)[label.long()].cuda()
        img_y = torch.cat([img, one_hot_y], dim=1)
        return self.D(img_y)

    def compute_loss(self):
        loss = None
        return loss


# class GAN(nn.Module):
#     def __init__(self, dim, W, H):
#         super(GAN, self).__init__()
#         self.dim = dim
#         self.G = Generator(dim, W, H)
#         self.D = Discriminator(W, H)
#
#     def forward(self, real):
#         B, _, _ = img.shape()
#         z = torch.randn(B, self.dim)
#         fake = self.G(z)
#         fake_score = self.D(fake)
#         real_score = self.G(real)
#         return None
#
#     def G_loss(self):
#         return None
#
#     def D_loss(self):
#         return None

