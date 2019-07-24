import torch.nn as nn


class ConvAutoencoder(nn.Module):
    def __init__(self, Cin):
        super(ConvAutoencoder, self).__init__()
        self.conv1 = nn.Conv2d(Cin, 16, 3)
        self.relu1 = nn.ReLU(inplace=True)
        self.maxpool1 = nn.MaxPool2d(2, return_indices=True)

        self.conv2 = nn.Conv2d(16, 32, 2)
        self.relu2 = nn.ReLU(inplace=True)
        self.maxpool2 = nn.MaxPool2d(2, return_indices=True)

        self.maxunpool1 = nn.MaxUnpool2d(2)
        self.relu3 = nn.ReLU(inplace=True)
        self.deconv1 = nn.ConvTranspose2d(32, 16, 2)

        self.maxunpool2 = nn.MaxUnpool2d(2)
        self.relu4 = nn.ReLU(inplace=True)
        self.deconv2 = nn.ConvTranspose2d(16, Cin, 3)

        self.loss_function = nn.MSELoss(size_average=False)

    def forward(self, x, label=None):
        encode1 = self.conv1(x)
        encode2, indices1 = self.maxpool1(self.relu1(encode1))
        encode3 = self.conv2(encode2)
        encode4, indices2 = self.maxpool2(self.relu2(encode3))
        decode1 = self.maxunpool1(encode4, indices2, output_size=encode3.size())
        decode2 = self.deconv1(self.relu3(decode1))
        decode3 = self.maxunpool2(decode2, indices1, output_size=encode1.size())
        decode4 = self.deconv2(self.relu4(decode3))
        return encode4, decode4

    def compute_loss(self, y_pred, y_true):
        return self.loss_function(y_pred, y_true)