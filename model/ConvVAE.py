import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable


def sampling(mean, log_var):
    eps = Variable(torch.randn(mean.shape)).cuda()
    return mean + eps * torch.exp(log_var / 2)


class ConvVAE(nn.Module):
    def __init__(self, Cin, latent_dim):
        super(ConvVAE, self).__init__()

        self.conv1 = nn.Conv2d(Cin, 16, 3)
        self.relu1 = nn.ReLU(inplace=True)
        self.maxpool1 = nn.MaxPool2d(2, return_indices=True)

        self.conv2 = nn.Conv2d(16, 32, 2)
        self.relu2 = nn.ReLU(inplace=True)
        self.maxpool2 = nn.MaxPool2d(2, return_indices=True)

        self.conv3 = nn.Conv2d(32, 64, 2)
        self.relu3 = nn.ReLU(inplace=True)
        self.maxpool3 = nn.MaxPool2d(2, return_indices=True)

        self.W = 9
        self.H = 7
        self.C = 64

        self.linear_mean = nn.Linear(self.C*self.W*self.H, latent_dim)
        self.linear_log_var = nn.Linear(self.C*self.W*self.H, latent_dim)
        self.FC = nn.Linear(latent_dim, self.C*self.W*self.H)

        self.maxunpool1 = nn.MaxUnpool2d(2)
        self.relu4 = nn.ReLU(inplace=True)
        self.deconv1 = nn.ConvTranspose2d(64, 32, 2)

        self.maxunpool2 = nn.MaxUnpool2d(2)
        self.relu5 = nn.ReLU(inplace=True)
        self.deconv2 = nn.ConvTranspose2d(32, 16, 2)

        self.maxunpool3 = nn.MaxUnpool2d(2)
        self.relu6 = nn.ReLU(inplace=True)
        self.deconv3 = nn.ConvTranspose2d(16, Cin, 3)
        self.norm = nn.Sigmoid()

        self.kl_loss = None
        self.loss_function = nn.MSELoss(size_average=False)

    def compute_loss(self, y_pre, y_true):
        # loss = self.loss_function(y_pre, y_true)
        # return loss + self.kl_loss
        return F.binary_cross_entropy(y_pre, y_true, size_average=False) + self.kl_loss, self.kl_loss

    def generator(self, x):
        return self.decoder(x)

    def forward(self, x, label=None):
        encode1 = self.conv1(x)
        encode2, indices1 = self.maxpool1(self.relu1(encode1))
        encode3 = self.conv2(encode2)
        encode4, indices2 = self.maxpool2(self.relu2(encode3))
        encode5 = self.conv3(encode4)
        encode6, indices3 = self.maxpool3(self.relu3(encode5))
        encode6 = encode6.reshape(-1, self.C*self.W*self.H)

        mean = self.linear_mean(encode6)
        log_var = self.linear_log_var(encode6)
        z = sampling(mean, log_var)
        self.kl_loss = -0.5 * torch.sum(1 + log_var - mean.pow(2) - log_var.exp())

        encode7 = self.FC(z).reshape(-1, self.C, self.W, self.H)
        decode1 = self.maxunpool1(encode7, indices3, output_size=encode5.size())
        decode2 = self.deconv1(self.relu4(decode1))
        decode3 = self.maxunpool2(decode2, indices2, output_size=encode3.size())
        decode4 = self.deconv2(self.relu5(decode3))
        decode5 = self.maxunpool3(decode4, indices1, output_size=encode1.size())
        decode6 = self.norm(self.deconv3(self.relu6(decode5)))
        return encode6, decode6