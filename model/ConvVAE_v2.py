import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable


class ConvVAE_v2(nn.Module):
    def __init__(self, W, H):
        super().__init__()

        intermediate_dim = 64
        latent_dim = 10

        img_channel = 1
        conv_features = (20, 30)
        self.conv_features = conv_features
        self.W = W
        self.H = H
        self.P = True

        self.encode_conv1 = nn.Conv2d(in_channels=img_channel,
                                      out_channels=conv_features[0],
                                      kernel_size=3,
                                      padding=self.P)

        self.encode_max_pool = nn.MaxPool2d(kernel_size=2, return_indices=True)

        self.encode_conv2 = nn.Conv2d(in_channels=conv_features[0],
                                      out_channels=conv_features[1],
                                      kernel_size=3,
                                      padding=self.P)

        self.total_pooled_size = conv_features[1] * (W // 4) * (H // 4)

        self.encode_fc = nn.Linear(in_features=self.total_pooled_size,
                                   out_features=intermediate_dim)

        self.linear_mean = nn.Linear(intermediate_dim, latent_dim)
        self.linear_log_var = nn.Linear(intermediate_dim, latent_dim)

        self.decode_fc1 = nn.Linear(in_features=latent_dim,
                                    out_features=intermediate_dim)

        self.decode_fc2 = nn.Linear(in_features=intermediate_dim,
                                    out_features=self.total_pooled_size)

        self.decode_unpool1 = nn.MaxUnpool2d(kernel_size=2)

        self.decode_unconv1 = nn.ConvTranspose2d(in_channels=conv_features[1],
                                                 out_channels=conv_features[0],
                                                 kernel_size=3,
                                                 padding=self.P)

        self.decode_unpool2 = nn.MaxUnpool2d(kernel_size=2)

        self.decode_unconv2 = nn.ConvTranspose2d(in_channels=conv_features[0],
                                                 out_channels=img_channel,
                                                 kernel_size=3,
                                                 padding=self.P)

        # self.decode_final_conv = nn.Conv2d(in_channels=img_channel,
        #                                    out_channels=img_channel,
        #                                    kernel_size=3,
        #                                    padding=True)

        self.kl_loss = None
        self.loss_function = nn.MSELoss(size_average=False)

    def compute_loss(self, y_pre, y_true):
        # return F.binary_cross_entropy(y_pre, y_true, size_average=False) + self.kl_loss
        return self.loss_function(y_pre, y_true) + self.kl_loss

    def decoder(self, z):
        z = F.leaky_relu(self.decode_fc1(z))
        z = F.leaky_relu(self.decode_fc2(z))
        z = z.view(-1, self.conv_features[1], self.W // 4, self.H // 4)
        z = self.decode_unpool1(z, indices=self.indices2, output_size=self.shape_2)
        z = F.leaky_relu(self.decode_unconv1(z))
        z = self.decode_unpool2(z, indices=self.indices1, output_size=self.shape_1)
        z = F.leaky_relu(self.decode_unconv2(z))
        # z = F.leaky_relu(self.decode_final_conv(z))
        # z = z.view(-1, self.W*self.H)
        return z

    def generator(self, x):
        return self.decoder(x)

    def sampling(self, mean, log_var):
        eps = Variable(torch.randn(mean.shape)).cuda()
        return mean + eps * torch.exp(log_var / 2)

    def forward(self, x, label=None):
        x = F.leaky_relu(self.encode_conv1(x))
        self.shape_1 = x.shape
        (x, self.indices1) = self.encode_max_pool(x)
        x = F.leaky_relu(self.encode_conv2(x))
        self.shape_2 = x.shape
        (x, self.indices2) = self.encode_max_pool(x)
        x = x.view(-1, self.total_pooled_size)
        x = F.leaky_relu(self.encode_fc(x))

        # compute mean and log_variance for z
        mean = self.linear_mean(x)
        log_var = self.linear_log_var(x)

        # sample z
        z = self.sampling(mean, log_var)

        # kl_loss
        # self.kl_loss = -0.5 * (1 + log_var - mean.pow(2) - log_var.exp()).sum()

        self.kl_loss = -0.5 * torch.sum(1 + log_var - mean.pow(2) - log_var.exp())

        return mean, self.decoder(z)